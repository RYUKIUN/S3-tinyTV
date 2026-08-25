/*
 * touch.cpp — XPT2046 → PC mouse, on the display's own SPI bus.
 *
 * DESIGN NOTES (why this is shaped the way it is)
 * ───────────────────────────────────────────────
 * 1. The display always wins the bus.
 *    displayTask holds g_spiMutex across its entire async DMA push. This task
 *    takes the same mutex around a single 57-byte read. Because g_spiMutex is
 *    a real mutex, FreeRTOS priority inheritance applies: when displayTask
 *    (prio 2) blocks on a bus held by this task (prio 1), this task is boosted
 *    to 2 and completes its read immediately. Worst case a frame is delayed by
 *    one XPT2046 read — 57 bytes at 1 MHz, about 456 us — and only while a
 *    finger is actually down. At the 30 Hz report rate that is ~1.4% bus
 *    occupancy during a touch and exactly 0% otherwise.
 *
 * 2. Idle cost depends on whether PENIRQ is wired (TOUCH_USE_IRQ in nexus.h).
 *    WIRED (1): PENIRQ goes low the moment the panel is pressed. This task
 *      blocks on a semaphore given by that pin's falling-edge ISR, so an
 *      untouched panel produces no wakeups, no SPI traffic and no CPU time at
 *      all. The ISR is detached while a touch is in progress (PENIRQ stays low
 *      throughout, which would otherwise storm us) and re-attached on release.
 *    UNWIRED (0, current): there is no cheap way to ask "is it touched", so we
 *      poll at TOUCH_IDLE_POLL_HZ and let the pressure reading answer. Costs
 *      one full read (~456 us) per poll — about 0.9% bus occupancy at 20 Hz —
 *      and delays touch-down by up to one poll period. Everything downstream
 *      is identical; only the wakeup source changes.
 *
 * 3. No filtering here.
 *    LovyanGFX's Touch_XPT2046::getTouchRaw() already takes 7 hardware samples
 *    per read and returns the median of each axis, and it checks PENIRQ before
 *    it touches the bus at all. Re-filtering on top of that would only add lag.
 *
 * 4. This task never calls sendto().
 *    Events go onto touchQueue; networkTask drains it. The UDP socket stays
 *    owned by exactly one task, so there is no concurrent-socket question and
 *    this task never blocks inside the network stack while holding the bus.
 *
 * 5. Gestures are NOT decided here.
 *    The ESP reports DOWN / MOVE / UP with panel coordinates and nothing more.
 *    Tap-vs-slide, thresholds and scroll feel all live in captureJpeg.py, where
 *    they can be retuned without reflashing.
 */
#include "touch.h"
#include "display.h"
#include <Preferences.h>

// ─────────────────────────────────────────────
//  CALIBRATION MODEL
// ─────────────────────────────────────────────
// We do NOT use LovyanGFX's calibrateTouch()/setTouchCalibrate() any more.
// Two reasons:
//
//   1. It samples exactly 4 corners. With no redundancy, every tap's error
//      lands directly in the transform — one sloppy corner skews the whole
//      panel. Here we take a TOUCH_CAL_GRID^2 grid and least-squares fit it,
//      so individual tap error averages out instead of accumulating.
//
//   2. Its transform is strictly affine (LovyanGFX stores 6 floats and there
//      is no way to hand it anything richer). Affine can express scale,
//      rotation, translation and shear, but NOT the gentle "twist" that
//      resistive panels genuinely have, where the x reading drifts with y and
//      vice versa. That twist is exactly what makes a 4-corner calibration
//      feel accurate near the corners and off toward the middle of the edges.
//
// So the model here is BILINEAR — one extra cross term per axis:
//
//     sx = a0 + a1*u + a2*v + a3*u*v
//     sy = b0 + b1*u + b2*v + b3*u*v      where u,v are the raw ADC / 4096
//
// It reduces to the affine case when a3/b3 fit to ~0, so it can only match or
// beat the old behaviour. Raw values are normalised to 0..1 before fitting
// because the u*v term would otherwise span ~1.6e7 next to a constant term of
// 1, which wrecks the conditioning of the normal equations in float.
//
// Because we fit raw -> final screen coordinates directly, panel rotation is
// baked into the fit and there is no convention to get wrong.
#define TOUCH_CAL_MAGIC   0x9341
#define TOUCH_CAL_VERSION 2   // v1 was LovyanGFX's 4-corner uint16_t[8]

#define TOUCH_CAL_TERMS   4   // 1, u, v, u*v
#define TOUCH_CAL_NPOINTS (TOUCH_CAL_GRID * TOUCH_CAL_GRID)

struct TouchCal {
    uint16_t magic;
    uint16_t version;
    float    cx[TOUCH_CAL_TERMS];   // raw -> screen X
    float    cy[TOUCH_CAL_TERMS];   // raw -> screen Y
};

static TouchCal s_cal;

// Basis vector for one raw sample. Kept in one place so the fit and the
// runtime mapping can never disagree about term order.
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

// ── BOOT-button double-press detector ────────────────────────────────────────
// Armed only before streaming starts, exactly as specified: once the stream is
// live the button does nothing, so there's no way to fumble into a
// recalibration mid-session.
#define BOOT_DOUBLE_PRESS_MS 600
static volatile uint32_t s_bootPressMs    = 0;
static volatile bool     s_recalRequested = false;

#if TOUCH_USE_IRQ
static void IRAM_ATTR touchIsr() {
    BaseType_t hpw = pdFALSE;
    xSemaphoreGiveFromISR(s_touchIrqSem, &hpw);
    if (hpw) portYIELD_FROM_ISR();
}
#endif

static void IRAM_ATTR bootIsr() {
    // Ignore the button entirely once the stream is up.
    if (g_streaming) return;
    uint32_t now = millis();
    // Crude debounce: anything under 40 ms is switch bounce, not a press.
    if (now - s_bootPressMs < 40) return;
    if (s_bootPressMs != 0 && (now - s_bootPressMs) <= BOOT_DOUBLE_PRESS_MS) {
        s_recalRequested = true;
        s_bootPressMs    = 0;
    } else {
        s_bootPressMs = now;
    }
}

// ─────────────────────────────────────────────
//  CALIBRATION PERSISTENCE
// ─────────────────────────────────────────────
static bool loadCalibration(TouchCal& cal) {
    Preferences prefs;
    if (!prefs.begin("touch", true)) return false;
    bool ok = false;
    if (prefs.getBytesLength("cal") == sizeof(TouchCal)) {
        prefs.getBytes("cal", &cal, sizeof(TouchCal));
        ok = (cal.magic == TOUCH_CAL_MAGIC && cal.version == TOUCH_CAL_VERSION);
    }
    prefs.end();
    return ok;
}

static void saveCalibration(const TouchCal& cal) {
    Preferences prefs;
    if (!prefs.begin("touch", false)) return;
    prefs.putBytes("cal", &cal, sizeof(TouchCal));
    prefs.end();
}

// Solve an n x n linear system in place by Gaussian elimination with partial
// pivoting. Doubles because this runs exactly once, at calibration time, and
// the normal equations are the one place precision actually matters.
static bool solveLinear(double A[TOUCH_CAL_TERMS][TOUCH_CAL_TERMS],
                        double b[TOUCH_CAL_TERMS], int n) {
    for (int col = 0; col < n; col++) {
        int piv = col;
        for (int r = col + 1; r < n; r++)
            if (fabs(A[r][col]) > fabs(A[piv][col])) piv = r;
        if (fabs(A[piv][col]) < 1e-12) return false;   // singular / degenerate taps

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

// One target: draw it, wait for a press, gather TOUCH_CAL_SAMPLES raw reads,
// return the median of each axis, then wait for release.
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
                // Drop the first reads: contact resistance is still settling as
                // the finger lands, and those samples are the least trustworthy
                // of the entire press.
                seen = true;
                delay(60);
                continue;
            }
            xs[got] = rx;
            ys[got] = ry;
            got++;
        } else if (seen) {
            seen = false;   // lifted early - start this target over
            got  = 0;
        }
        delay(8);
    }

    // Median, not mean: robust against a rogue sample in a way an average
    // simply is not.
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
    while (lcd.getTouchRaw(&dx, &dy)) delay(10);   // wait for release
    delay(120);
}

// Collect the grid, fit it, and report how good the fit actually is.
// Returns the RMS residual in pixels, or a large value if the solve failed.
static float calibrateOnce() {
    const int N = TOUCH_CAL_NPOINTS;
    int32_t rawX[TOUCH_CAL_NPOINTS], rawY[TOUCH_CAL_NPOINTS];
    int     tgtX[TOUCH_CAL_NPOINTS], tgtY[TOUCH_CAL_NPOINTS];

    const int span_x = SCREEN_W - 1 - 2 * TOUCH_CAL_INSET;
    const int span_y = SCREEN_H - 1 - 2 * TOUCH_CAL_INSET;

    int idx = 0;
    for (int gy = 0; gy < TOUCH_CAL_GRID; gy++) {
        for (int gx = 0; gx < TOUCH_CAL_GRID; gx++) {
            // Serpentine order: every target is adjacent to the previous one,
            // so the finger takes the shortest path across the whole run.
            int col = (gy & 1) ? (TOUCH_CAL_GRID - 1 - gx) : gx;
            tgtX[idx] = TOUCH_CAL_INSET + span_x * col / (TOUCH_CAL_GRID - 1);
            tgtY[idx] = TOUCH_CAL_INSET + span_y * gy  / (TOUCH_CAL_GRID - 1);
            idx++;
        }
    }

    for (int i = 0; i < N; i++)
        sampleTarget(tgtX[i], tgtY[i], i + 1, N, rawX[i], rawY[i]);

    // Normal equations for both axes at once: they share a design matrix and
    // differ only in the right-hand side.
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

    // solveLinear destroys its matrix, so each axis gets its own copy.
    double Ax[TOUCH_CAL_TERMS][TOUCH_CAL_TERMS], Ay[TOUCH_CAL_TERMS][TOUCH_CAL_TERMS];
    memcpy(Ax, ATA, sizeof(ATA));
    memcpy(Ay, ATA, sizeof(ATA));
    if (!solveLinear(Ax, ATx, TOUCH_CAL_TERMS)) return 9999.0f;
    if (!solveLinear(Ay, ATy, TOUCH_CAL_TERMS)) return 9999.0f;

    for (int i = 0; i < TOUCH_CAL_TERMS; i++) {
        s_cal.cx[i] = (float)ATx[i];
        s_cal.cy[i] = (float)ATy[i];
    }

    // Residual: how far the fitted model lands from the targets we actually
    // tapped. This is the number that says whether it worked.
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

        // A bad fit almost always means one tap landed well off its marker.
        // Saving it would degrade every touch from here on, and we can already
        // measure that it is wrong - so redo the run instead of persisting it.
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
    saveCalibration(s_cal);
    g_touchCalibrated = true;

    Serial.printf("[TOUCH] Calibrated: %d points, RMS %.2f px\n", TOUCH_CAL_NPOINTS, rms);

    lcd.fillScreen(TFT_BLACK);
    lcd.setTextColor(rms <= TOUCH_CAL_MAX_RMS ? TFT_GREEN : TFT_ORANGE, TFT_BLACK);
    lcd.drawString("Calibration saved.", 8, 90);
    lcd.setTextColor(0x7BEF, TFT_BLACK);
    {
        char msg[40];
        snprintf(msg, sizeof(msg), "accuracy: %.1f px RMS", rms);
        lcd.drawString(msg, 8, 112);
    }
    delay(1400);
}

void touchBegin() {
#if TOUCH_USE_IRQ
    s_touchIrqSem = xSemaphoreCreateBinary();

    // LovyanGFX's touch init leaves PENIRQ as a plain input. XPT2046 drives it
    // actively low, so a pull-up costs nothing and keeps the line from floating
    // (and firing phantom interrupts) on modules without one fitted.
    pinMode(TOUCH_PIN_IRQ, INPUT_PULLUP);
#else
    // PENIRQ is not wired. Deliberately left completely alone — no pinMode, no
    // pull-up, no interrupt. If the line is shorted rather than merely floating,
    // a pull-up would just sink current into the short for no benefit.
#endif

    // The BOOT button is armed here so the double-press window covers the whole
    // pre-stream period: boot screen, WiFi association, and "Waiting for PC...".
    pinMode(BOOT_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(BOOT_PIN), bootIsr, FALLING);

    // A v1 blob (LovyanGFX's 4-corner uint16_t[8]) fails the version check and
    // falls through to a fresh run — which is what we want, since the whole
    // point of v2 is that the old fit was the thing being inaccurate.
    if (loadCalibration(s_cal)) {
        g_touchCalibrated = true;
        Serial.println("[TOUCH] Calibration loaded from NVS (bilinear, v2)");
    } else {
        Serial.println("[TOUCH] No usable calibration — running setup");
        runCalibration();
    }
}

// ─────────────────────────────────────────────
//  SAMPLER TASK
// ─────────────────────────────────────────────
// Reads one touch point under the bus mutex. Returns false both when nothing is
// being touched AND when the display owned the bus for longer than we were
// willing to wait — the caller distinguishes the two via `gotBus`, because a
// skipped sample must never be mistaken for a release.
static bool readPoint(int32_t& x, int32_t& y, bool& gotBus) {
    gotBus = (xSemaphoreTake(g_spiMutex, pdMS_TO_TICKS(TOUCH_BUS_WAIT_MS)) == pdTRUE);
    if (!gotBus) return false;
    // getTouchRaw, not getTouch: we want the unmapped ADC reading so our own
    // bilinear fit can do the mapping. getTouch() would apply LovyanGFX's
    // affine transform, which is the one we deliberately replaced.
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
    // Never block: touch is strictly best-effort next to the video stream. If
    // the queue is full then networkTask is backed up, and a stale MOVE is
    // worth less than the delay that waiting for it would cost.
    xQueueSend(touchQueue, &ev, 0);
}

void touchTask(void*) {
    uint8_t touchId  = 0;
    uint8_t seq      = 0;
    bool    pressed  = false;
    uint8_t emptyRun = 0;   // consecutive empty reads, debounces release
    uint8_t pressRun = 0;   // consecutive valid reads, debounces press
    uint32_t lastSendMs = 0;
    int32_t lastX = 0, lastY = 0;

    // Two rates: report rate while a finger is down, and a slower idle rate
    // used only when we have to poll for the press in the first place.
    const TickType_t samplePeriod = pdMS_TO_TICKS(1000 / TOUCH_REPORT_HZ);
    const TickType_t idlePeriod   = pdMS_TO_TICKS(1000 / TOUCH_IDLE_POLL_HZ);

    for (;;) {
        if (!pressed) {
            // Pre-stream, we look for a BOOT double-press. Once streaming
            // starts the button is dead (bootIsr bails on g_streaming).
            if (!g_streaming && s_recalRequested) {
                s_recalRequested = false;
#if TOUCH_USE_IRQ
                detachInterrupt(digitalPinToInterrupt(TOUCH_PIN_IRQ));
                s_irqAttached = false;
#endif
                // Hold the bus for the whole interactive run: wifiWatchdogTask
                // is still repainting the WiFi status line behind us.
                xSemaphoreTake(g_spiMutex, portMAX_DELAY);
                runCalibration();
                xSemaphoreGive(g_spiMutex);
                Serial.println("[TOUCH] Recalibrated via BOOT double-press");
                continue;
            }

#if TOUCH_USE_IRQ
            // Idle: sleep on PENIRQ. Zero CPU, zero SPI, until a finger lands.
            if (!s_irqAttached) {
                attachInterrupt(digitalPinToInterrupt(TOUCH_PIN_IRQ), touchIsr, FALLING);
                s_irqAttached = true;
            }

            // Pre-stream we wake periodically so the BOOT check above still
            // runs; once streaming, sleep indefinitely and cost nothing.
            TickType_t idleWait = g_streaming ? portMAX_DELAY : pdMS_TO_TICKS(200);
            if (xSemaphoreTake(s_touchIrqSem, idleWait) != pdTRUE) continue;

            // PENIRQ stays LOW for the whole press, so leaving the edge
            // interrupt attached would storm us. Detach and switch to polling.
            detachInterrupt(digitalPinToInterrupt(TOUCH_PIN_IRQ));
            s_irqAttached = false;
#endif
            // Without PENIRQ there's nothing to wait on — fall straight through
            // and let the pressure reading below decide. The delay at the
            // bottom of the loop paces it at TOUCH_IDLE_POLL_HZ.
        }

        int32_t x = 0, y = 0;
        bool    gotBus  = false;
        bool    touched = readPoint(x, y, gotBus);

        if (!gotBus) {
            // Display held the bus past our patience. Not a release — just try
            // again next tick without disturbing the press state machine.
            vTaskDelay(pressed ? samplePeriod : idlePeriod);
            continue;
        }

        if (touched) {
            emptyRun = 0;
            if (!pressed) {
                // Require TOUCH_PRESS_SAMPLES corroborating reads before
                // committing. At the default of 1 this fires immediately.
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
                // Suppress sub-pixel jitter: an unmoved finger sends nothing —
                // except for the periodic keepalive, which is what lets the PC
                // distinguish a deliberately still finger from a dead link.
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
            pressRun = 0;   // isolated blip, not the start of a press
        } else {
            // Resistive panels drop a sample or two mid-press; require a run of
            // empty reads before calling it a release.
            pressRun = 0;
            if (++emptyRun >= TOUCH_RELEASE_SAMPLES) {
                pressed  = false;
                emptyRun = 0;
                postEvent(TOUCH_EV_UP, touchId, seq++, lastX, lastY);
                continue;   // straight back to the idle path
            }
        }

        vTaskDelay(pressed ? samplePeriod : idlePeriod);
    }
}
