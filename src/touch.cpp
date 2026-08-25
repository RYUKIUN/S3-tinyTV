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

// Calibration blob: LovyanGFX's calibrateTouch() emits uint16_t[8] (four
// screen corners' raw ADC readings) which setTouchCalibrate() turns back into
// the affine transform. Versioned so a format change can't be misread as valid.
#define TOUCH_CAL_MAGIC   0x9341
#define TOUCH_CAL_VERSION 1

struct TouchCal {
    uint16_t magic;
    uint16_t version;
    uint16_t params[8];
};

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

static void runCalibration() {
    TouchCal cal = { TOUCH_CAL_MAGIC, TOUCH_CAL_VERSION, {0} };

    lcd.fillScreen(TFT_BLACK);
    lcd.setTextFont(2);
    lcd.setTextSize(1);
    lcd.setTextColor(TFT_WHITE, TFT_BLACK);
    lcd.drawString("TOUCH CALIBRATION", 8, 88);
    lcd.setTextColor(0x7BEF, TFT_BLACK);
    lcd.drawString("Tap each corner marker.", 8, 110);
    delay(1200);

    // Blocking and interactive, but this only ever runs pre-WiFi on a first
    // boot or after an explicit BOOT double-press — never while streaming.
    lcd.calibrateTouch(cal.params, TFT_WHITE, TFT_BLACK, 15);

    saveCalibration(cal);
    g_touchCalibrated = true;

    lcd.fillScreen(TFT_BLACK);
    lcd.setTextColor(TFT_GREEN, TFT_BLACK);
    lcd.drawString("Calibration saved.", 8, 100);
    delay(900);
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

    TouchCal cal;
    if (loadCalibration(cal)) {
        lcd.setTouchCalibrate(cal.params);
        g_touchCalibrated = true;
        Serial.println("[TOUCH] Calibration loaded from NVS");
    } else {
        Serial.println("[TOUCH] No stored calibration — running setup");
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
    bool touched = lcd.getTouch(&x, &y);
    xSemaphoreGive(g_spiMutex);
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
                postEvent(TOUCH_EV_DOWN, touchId, seq++, x, y);
            } else if (abs((int)x - (int)lastX) >= TOUCH_MOVE_EPS ||
                       abs((int)y - (int)lastY) >= TOUCH_MOVE_EPS) {
                // Suppress sub-pixel jitter: an unmoved finger sends nothing.
                lastX = x;
                lastY = y;
                postEvent(TOUCH_EV_MOVE, touchId, seq++, x, y);
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
