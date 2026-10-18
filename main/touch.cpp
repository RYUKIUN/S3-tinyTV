/*
 * touch.cpp — XPT2046 → PC mouse, on the display's own SPI bus.
 *
 * (See nexus.h Zone 1 for the bus-sharing/PENIRQ design notes — unchanged
 * from the original. The only things this rebase touched are the three
 * marked REBASE blocks below: NVS calibration storage replacing
 * Preferences, and native driver/gpio.h replacing Arduino's pinMode/
 * attachInterrupt/digitalPinToInterrupt/FALLING/INPUT_PULLUP.)
 */
#include "nexus.h"

#if TOUCH_FEATURE_ENABLED
#include "touch.h"
#include "display.h"
#include "nvs.h"
#include "driver/gpio.h"
#include "esp_log.h"

static const char* TAG = "touch";

#define TOUCH_CAL_MAGIC   0x9341
#define TOUCH_CAL_VERSION 2

#define TOUCH_CAL_TERMS   4   // 1, u, v, u*v
#define TOUCH_CAL_NPOINTS (TOUCH_CAL_GRID * TOUCH_CAL_GRID)

struct TouchCal {
    uint16_t magic;
    uint16_t version;
    float    cx[TOUCH_CAL_TERMS];   // raw -> screen X
    float    cy[TOUCH_CAL_TERMS];   // raw -> screen Y
};

static TouchCal s_cal;

static inline void calBasis(int32_t rawX, int32_t rawY, float* t) {
    float u = (float)rawX * (1.0f / 4096.0f);
    float v = (float)rawY * (1.0f / 4096.0f);
    t[0] = 1.0f;
    t[1] = u;
    t[2] = v;
    t[3] = u * v;
}

static inline void calApply(int32_t rawX, int32_t rawY, int32_t& sx, int32_t& sy) {
    float t[TOUCH_CAL_TERMS];
    calBasis(rawX, rawY, t);
    float fx = 0.0f, fy = 0.0f;
    for (int i = 0; i < TOUCH_CAL_TERMS; i++) {
        fx += s_cal.cx[i] * t[i];
        fy += s_cal.cy[i] * t[i];
    }
    sx = (int32_t)lroundf(fx);
    sy = (int32_t)lroundf(fy);
}

#if TOUCH_USE_IRQ
static SemaphoreHandle_t s_touchIrqSem = nullptr;
static bool              s_irqAttached = false;
#endif

#define BOOT_DOUBLE_PRESS_MS 600
static volatile uint32_t s_bootPressMs    = 0;
static volatile bool     s_recalRequested = false;

#if TOUCH_USE_IRQ
static void IRAM_ATTR touchIsr(void* /*arg*/) {
    BaseType_t hpw = pdFALSE;
    xSemaphoreGiveFromISR(s_touchIrqSem, &hpw);
    if (hpw) portYIELD_FROM_ISR();
}
#endif

static void IRAM_ATTR bootIsr(void* /*arg*/) {
    if (g_streaming) return;
    uint32_t now = millis();
    if (now - s_bootPressMs < 40) return;
    if (s_bootPressMs != 0 && (now - s_bootPressMs) <= BOOT_DOUBLE_PRESS_MS) {
        s_recalRequested = true;
        s_bootPressMs    = 0;
    } else {
        s_bootPressMs = now;
    }
}

// ─────────────────────────────────────────────
//  CALIBRATION PERSISTENCE  [REBASE: nvs.h replaces Preferences]
// ─────────────────────────────────────────────
// Preferences (Arduino-ESP32) is a thin wrapper over exactly this NVS API —
// swapping it out is a 1:1 mechanical change, same "touch" namespace, same
// blob key, same bytes on flash. A calibration saved by the old Arduino
// build is still readable here (and vice versa).
static bool loadCalibration(TouchCal& cal) {
    nvs_handle_t h;
    esp_err_t err = nvs_open("touch", NVS_READONLY, &h);
    if (err == ESP_ERR_NVS_NOT_FOUND) return false;
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "failed to open calibration storage: %s", esp_err_to_name(err));
        return false;
    }

    size_t len = sizeof(TouchCal);
    err = nvs_get_blob(h, "cal", &cal, &len);
    nvs_close(h);

    if (err != ESP_OK && err != ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGE(TAG, "failed to read calibration: %s", esp_err_to_name(err));
        return false;
    }
    if (err != ESP_OK || len != sizeof(TouchCal)) return false;
    return (cal.magic == TOUCH_CAL_MAGIC && cal.version == TOUCH_CAL_VERSION);
}

static bool saveCalibration(const TouchCal& cal) {
    nvs_handle_t h;
    esp_err_t err = nvs_open("touch", NVS_READWRITE, &h);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "failed to open calibration storage: %s", esp_err_to_name(err));
        return false;
    }

    err = nvs_set_blob(h, "cal", &cal, sizeof(TouchCal));
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "failed to save calibration: %s", esp_err_to_name(err));
        return false;
    }
    return true;
}

static bool solveLinear(double A[TOUCH_CAL_TERMS][TOUCH_CAL_TERMS],
                        double b[TOUCH_CAL_TERMS], int n) {
    for (int col = 0; col < n; col++) {
        int piv = col;
        for (int r = col + 1; r < n; r++)
            if (fabs(A[r][col]) > fabs(A[piv][col])) piv = r;
        if (fabs(A[piv][col]) < 1e-12) return false;

        if (piv != col) {
            for (int c = 0; c < n; c++) { double t = A[col][c]; A[col][c] = A[piv][c]; A[piv][c] = t; }
            double t = b[col]; b[col] = b[piv]; b[piv] = t;
        }
        for (int r = col + 1; r < n; r++) {
            double f = A[r][col] / A[col][col];
            if (f == 0.0) continue;
            for (int c = col; c < n; c++) A[r][c] -= f * A[col][c];
            b[r] -= f * b[col];
        }
    }
    for (int r = n - 1; r >= 0; r--) {
        double sum = b[r];
        for (int c = r + 1; c < n; c++) sum -= A[r][c] * b[c];
        b[r] = sum / A[r][r];
    }
    return true;
}

static void sampleTarget(int tx, int ty, int idx1, int total,
                         int32_t& outX, int32_t& outY) {
    const int R = 9;
    lcd.fillScreen(TFT_BLACK);
    lcd.setTextFont(2);
    lcd.setTextSize(1);
    lcd.setTextColor(0x7BEF, TFT_BLACK);
    lcd.drawString("Tap the marker centre", 8, 6);

    char prog[16];
    snprintf(prog, sizeof(prog), "%d/%d", idx1, total);
    lcd.setTextColor(TFT_WHITE, TFT_BLACK);
    lcd.drawString(prog, SCREEN_W - 44, 6);

    lcd.drawFastHLine(tx - R, ty, R * 2 + 1, TFT_WHITE);
    lcd.drawFastVLine(tx, ty - R, R * 2 + 1, TFT_WHITE);
    lcd.drawCircle(tx, ty, R, TFT_RED);

    int32_t xs[TOUCH_CAL_SAMPLES], ys[TOUCH_CAL_SAMPLES];
    int  got  = 0;
    bool seen = false;

    while (got < TOUCH_CAL_SAMPLES) {
        int32_t rx, ry;
        if (lcd.getTouchRaw(&rx, &ry)) {
            if (!seen) {
                seen = true;
                delay(60);
                continue;
            }
            xs[got] = rx;
            ys[got] = ry;
            got++;
        } else if (seen) {
            seen = false;
            got  = 0;
        }
        delay(8);
    }

    for (int i = 1; i < got; i++) {
        int32_t kx = xs[i], ky = ys[i];
        int j = i - 1;
        while (j >= 0 && xs[j] > kx) { xs[j + 1] = xs[j]; j--; }
        xs[j + 1] = kx;
        j = i - 1;
        while (j >= 0 && ys[j] > ky) { ys[j + 1] = ys[j]; j--; }
        ys[j + 1] = ky;
    }
    outX = xs[got / 2];
    outY = ys[got / 2];

    lcd.drawCircle(tx, ty, R, TFT_GREEN);

    int32_t dx, dy;
    while (lcd.getTouchRaw(&dx, &dy)) delay(10);
    delay(120);
}

static float calibrateOnce() {
    const int N = TOUCH_CAL_NPOINTS;
    int32_t rawX[TOUCH_CAL_NPOINTS], rawY[TOUCH_CAL_NPOINTS];
    int     tgtX[TOUCH_CAL_NPOINTS], tgtY[TOUCH_CAL_NPOINTS];

    const int span_x = SCREEN_W - 1 - 2 * TOUCH_CAL_INSET;
    const int span_y = SCREEN_H - 1 - 2 * TOUCH_CAL_INSET;

    int idx = 0;
    for (int gy = 0; gy < TOUCH_CAL_GRID; gy++) {
        for (int gx = 0; gx < TOUCH_CAL_GRID; gx++) {
            int col = (gy & 1) ? (TOUCH_CAL_GRID - 1 - gx) : gx;
            tgtX[idx] = TOUCH_CAL_INSET + span_x * col / (TOUCH_CAL_GRID - 1);
            tgtY[idx] = TOUCH_CAL_INSET + span_y * gy  / (TOUCH_CAL_GRID - 1);
            idx++;
        }
    }

    for (int i = 0; i < N; i++)
        sampleTarget(tgtX[i], tgtY[i], i + 1, N, rawX[i], rawY[i]);

    double ATA[TOUCH_CAL_TERMS][TOUCH_CAL_TERMS] = {};
    double ATx[TOUCH_CAL_TERMS] = {};
    double ATy[TOUCH_CAL_TERMS] = {};

    for (int i = 0; i < N; i++) {
        float t[TOUCH_CAL_TERMS];
        calBasis(rawX[i], rawY[i], t);
        for (int r = 0; r < TOUCH_CAL_TERMS; r++) {
            for (int c = 0; c < TOUCH_CAL_TERMS; c++) ATA[r][c] += (double)t[r] * t[c];
            ATx[r] += (double)t[r] * tgtX[i];
            ATy[r] += (double)t[r] * tgtY[i];
        }
    }

    double Ax[TOUCH_CAL_TERMS][TOUCH_CAL_TERMS], Ay[TOUCH_CAL_TERMS][TOUCH_CAL_TERMS];
    memcpy(Ax, ATA, sizeof(ATA));
    memcpy(Ay, ATA, sizeof(ATA));
    if (!solveLinear(Ax, ATx, TOUCH_CAL_TERMS)) return 9999.0f;
    if (!solveLinear(Ay, ATy, TOUCH_CAL_TERMS)) return 9999.0f;

    for (int i = 0; i < TOUCH_CAL_TERMS; i++) {
        s_cal.cx[i] = (float)ATx[i];
        s_cal.cy[i] = (float)ATy[i];
    }

    double acc = 0.0;
    for (int i = 0; i < N; i++) {
        int32_t sx, sy;
        calApply(rawX[i], rawY[i], sx, sy);
        double dx = (double)sx - tgtX[i];
        double dy = (double)sy - tgtY[i];
        acc += dx * dx + dy * dy;
    }
    return (float)sqrt(acc / N);
}

static void runCalibration() {
    lcd.fillScreen(TFT_BLACK);
    lcd.setTextFont(2);
    lcd.setTextSize(1);
    lcd.setTextColor(TFT_WHITE, TFT_BLACK);
    lcd.drawString("TOUCH CALIBRATION", 8, 78);
    lcd.setTextColor(0x7BEF, TFT_BLACK);
    {
        char msg[52];
        snprintf(msg, sizeof(msg), "%d points - a stylus helps if you have one.",
                 TOUCH_CAL_NPOINTS);
        lcd.drawString(msg, 8, 100);
    }
    lcd.drawString("Tap each marker centre precisely.", 8, 120);
    delay(2000);

    float rms = 9999.0f;
    for (int attempt = 0; attempt <= TOUCH_CAL_MAX_RETRY; attempt++) {
        rms = calibrateOnce();
        if (rms <= TOUCH_CAL_MAX_RMS) break;

        if (attempt < TOUCH_CAL_MAX_RETRY) {
            lcd.fillScreen(TFT_BLACK);
            lcd.setTextColor(TFT_ORANGE, TFT_BLACK);
            lcd.drawString("Fit looks off - redoing it.", 8, 88);
            lcd.setTextColor(0x7BEF, TFT_BLACK);
            char msg[44];
            snprintf(msg, sizeof(msg), "error %.1f px, want under %.0f", rms, TOUCH_CAL_MAX_RMS);
            lcd.drawString(msg, 8, 110);
            delay(2200);
        }
    }

    s_cal.magic   = TOUCH_CAL_MAGIC;
    s_cal.version = TOUCH_CAL_VERSION;
    bool saved = saveCalibration(s_cal);
    g_touchCalibrated = true;

    ESP_LOGI(TAG, "Calibrated: %d points, RMS %.2f px", TOUCH_CAL_NPOINTS, rms);

    lcd.fillScreen(TFT_BLACK);
    lcd.setTextColor(rms <= TOUCH_CAL_MAX_RMS ? TFT_GREEN : TFT_ORANGE, TFT_BLACK);
    lcd.drawString(saved ? "Calibration saved." : "Save failed - check log.",
                   8, 90);
    lcd.setTextColor(0x7BEF, TFT_BLACK);
    {
        char msg[40];
        snprintf(msg, sizeof(msg), "accuracy: %.1f px RMS", rms);
        lcd.drawString(msg, 8, 112);
    }

    delay(1400);
}

void touchBegin() {
    // [REBASE: driver/gpio.h replaces pinMode/attachInterrupt/FALLING/
    // INPUT_PULLUP/digitalPinToInterrupt. gpio_install_isr_service() is the
    // ESP-IDF equivalent of Arduino's implicit "interrupts just work" setup —
    // it has to be called once before any gpio_isr_handler_add().]
    static bool isrServiceInstalled = false;
    if (!isrServiceInstalled) {
        gpio_install_isr_service(0);
        isrServiceInstalled = true;
    }

#if TOUCH_USE_IRQ
    s_touchIrqSem = xSemaphoreCreateBinary();

    gpio_config_t irqCfg = {};
    irqCfg.pin_bit_mask = (1ULL << TOUCH_PIN_IRQ);
    irqCfg.mode         = GPIO_MODE_INPUT;
    irqCfg.pull_up_en   = GPIO_PULLUP_ENABLE;
    irqCfg.intr_type    = GPIO_INTR_DISABLE;   // attached on demand, see touchTask
    gpio_config(&irqCfg);
#else
    // PENIRQ is not wired. Deliberately left completely alone.
#endif

    gpio_config_t bootCfg = {};
    bootCfg.pin_bit_mask = (1ULL << BOOT_PIN);
    bootCfg.mode         = GPIO_MODE_INPUT;
    bootCfg.pull_up_en   = GPIO_PULLUP_ENABLE;
    bootCfg.intr_type    = GPIO_INTR_NEGEDGE;   // FALLING
    gpio_config(&bootCfg);
    gpio_isr_handler_add((gpio_num_t)BOOT_PIN, bootIsr, nullptr);

    if (loadCalibration(s_cal)) {
        g_touchCalibrated = true;
        ESP_LOGI(TAG, "Calibration loaded from NVS (bilinear, v2)");
    } else {
        ESP_LOGI(TAG, "No usable calibration — running setup");
        runCalibration();
    }
}

static bool readPoint(int32_t& x, int32_t& y, bool& gotBus) {
    gotBus = (xSemaphoreTake(g_spiMutex, pdMS_TO_TICKS(TOUCH_BUS_WAIT_MS)) == pdTRUE);
    if (!gotBus) return false;
    int32_t rx, ry;
    bool touched = lcd.getTouchRaw(&rx, &ry);
    xSemaphoreGive(g_spiMutex);
    if (touched) calApply(rx, ry, x, y);
    return touched;
}

static void postEvent(uint8_t kind, uint8_t tid, uint8_t seq, int32_t x, int32_t y) {
    if (!g_touchEnabled) return;
    TouchEvent ev = { kind, tid, seq,
                      (uint16_t)constrain(x, 0, SCREEN_W - 1),
                      (uint16_t)constrain(y, 0, SCREEN_H - 1) };
    xQueueSend(touchQueue, &ev, 0);
}

void touchTask(void*) {
    uint8_t touchId  = 0;
    uint8_t seq      = 0;
    bool    pressed  = false;
    uint8_t emptyRun = 0;
    uint8_t pressRun = 0;
    uint32_t lastSendMs = 0;
    int32_t lastX = 0, lastY = 0;

    const TickType_t samplePeriod = pdMS_TO_TICKS(1000 / TOUCH_REPORT_HZ);
    const TickType_t idlePeriod   = pdMS_TO_TICKS(1000 / TOUCH_IDLE_POLL_HZ);

    for (;;) {
        if (!pressed) {
            if (!g_streaming && s_recalRequested) {
                s_recalRequested = false;
#if TOUCH_USE_IRQ
                gpio_isr_handler_remove((gpio_num_t)TOUCH_PIN_IRQ);
                s_irqAttached = false;
#endif
                xSemaphoreTake(g_spiMutex, portMAX_DELAY);
                runCalibration();
                xSemaphoreGive(g_spiMutex);
                ESP_LOGI(TAG, "Recalibrated via BOOT double-press");
                continue;
            }

#if TOUCH_USE_IRQ
            if (!s_irqAttached) {
                gpio_set_intr_type((gpio_num_t)TOUCH_PIN_IRQ, GPIO_INTR_NEGEDGE);
                gpio_isr_handler_add((gpio_num_t)TOUCH_PIN_IRQ, touchIsr, nullptr);
                s_irqAttached = true;
            }

            TickType_t idleWait = g_streaming ? portMAX_DELAY : pdMS_TO_TICKS(200);
            if (xSemaphoreTake(s_touchIrqSem, idleWait) != pdTRUE) continue;

            gpio_isr_handler_remove((gpio_num_t)TOUCH_PIN_IRQ);
            s_irqAttached = false;
#endif
        }

        int32_t x = 0, y = 0;
        bool    gotBus  = false;
        bool    touched = readPoint(x, y, gotBus);

        if (!gotBus) {
            vTaskDelay(pressed ? samplePeriod : idlePeriod);
            continue;
        }

        if (touched) {
            emptyRun = 0;
            if (!pressed) {
                if (++pressRun < TOUCH_PRESS_SAMPLES) {
                    vTaskDelay(idlePeriod);
                    continue;
                }
                pressRun = 0;
                pressed  = true;
                touchId++;
                seq   = 0;
                lastX = x;
                lastY = y;
                lastSendMs = millis();
                postEvent(TOUCH_EV_DOWN, touchId, seq++, x, y);
            } else {
                bool moved = (abs((int)x - (int)lastX) >= TOUCH_MOVE_EPS ||
                              abs((int)y - (int)lastY) >= TOUCH_MOVE_EPS);
                if (moved || (millis() - lastSendMs) >= TOUCH_KEEPALIVE_MS) {
                    lastX = x;
                    lastY = y;
                    lastSendMs = millis();
                    postEvent(TOUCH_EV_MOVE, touchId, seq++, x, y);
                }
            }
        } else if (!pressed) {
            pressRun = 0;
        } else {
            pressRun = 0;
            if (++emptyRun >= TOUCH_RELEASE_SAMPLES) {
                pressed  = false;
                emptyRun = 0;
                postEvent(TOUCH_EV_UP, touchId, seq++, lastX, lastY);
                continue;
            }
        }

        vTaskDelay(pressed ? samplePeriod : idlePeriod);
    }
}

#endif
