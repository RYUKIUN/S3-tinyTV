#include "display.h"
#include "esp_task_wdt.h"

LGFX lcd;

// Bus ownership helpers. Null-safe because drawBootHeader()/statusLine() run
// during early setup(), potentially before the mutex object exists — and a
// single-threaded boot needs no arbitration anyway.
//
// These matter beyond displayTask: statusLine() is called from decodeTask
// (Core 1, on first frame) and wifiWatchdogTask (Core 0, every 250 ms while
// not streaming). Those are short synchronous draws that LovyanGFX's own
// use_lock used to cover on its own, but Touch_XPT2046 issues its transfer
// through lgfx::spi::* directly rather than through the panel's bus object,
// so it sits outside that lock. One explicit mutex now covers every path that
// can touch the bus, from any task, on either core.
static inline void busLock()   { if (g_spiMutex) xSemaphoreTake(g_spiMutex, portMAX_DELAY); }
static inline void busUnlock() { if (g_spiMutex) xSemaphoreGive(g_spiMutex); }

void initDisplay() {
    lcd.init();
    lcd.setRotation(3);
    lcd.setColorDepth(16);
    lcd.setTextFont(2);
    lcd.setTextSize(1);
}

void statusLine(uint8_t row, const char* label, const char* value, uint32_t col) {
    int y = 58 + row * 22;
    busLock();
    lcd.startWrite();
    lcd.fillRect(0, y, SCREEN_W, 22, TFT_BLACK);
    lcd.setTextColor(0x7BEF, TFT_BLACK);
    lcd.drawString(label, 8, y + 3);
    lcd.setTextColor(col, TFT_BLACK);
    lcd.drawString(value, 138, y + 3);
    lcd.endWrite();
    busUnlock();
}

void drawBootHeader() {
    busLock();
    lcd.startWrite();
    lcd.fillScreen(TFT_BLACK);
    lcd.setTextFont(2);
    lcd.setTextSize(1);
    lcd.fillRect(0, 0, SCREEN_W, 54, 0x1082);
    lcd.setTextColor(TFT_CYAN, 0x1082);
    lcd.setTextSize(2);
    lcd.drawString("ESP32-S3 STREAM", 8, 6);
    lcd.setTextSize(1);
    lcd.setTextColor(0x7BEF, 0x1082);
    lcd.drawString("ILI9341  320x240  ping-pong", 8, 34);
    lcd.drawFastHLine(0, 54, SCREEN_W, TFT_DARKGREY);
    lcd.endWrite();
    busUnlock();
}

void displayTask(void*) {
    esp_task_wdt_add(NULL);  // register this task as its own TWDT subscriber

    DisplayMsg dmsg;
    bool dmaPending    = false;  // true while a pushPixelsDMA transfer is in flight
    bool overlayVisible = false;
    uint32_t lastOverlayMs = 0;

    while (true) {
        // ── Poll DMA completion — yield cooperatively while busy ──────────────
        // pushPixelsDMA returns immediately; we loop here with taskYIELD() so
        // the IDLE task (and other tasks) keep getting CPU time, which also feeds
        // the TWDT without any artificial vTaskDelay.
        if (dmaPending) {
            if (lcd.dmaBusy()) {
                esp_task_wdt_reset();
                taskYIELD();
                continue;
            }
            // DMA finished — release bus and account for the frame
            lcd.waitDMA();   // guaranteed near-instant since dmaBusy() was false
            lcd.endWrite();
            busUnlock();   // touch may have the bus now
            dmaPending = false;
            g_presentedFrames++;
            overlayVisible = false;
            esp_task_wdt_reset();
        }

        // ── Pull next frame from queue ────────────────────────────────────────
        // 8 ms timeout: tight enough to keep the DMA-busy polling loop responsive,
        // long enough not to burn CPU on empty spins between frames.
        bool gotFrame = (xQueueReceive(displayQueue, &dmsg, pdMS_TO_TICKS(8)) == pdTRUE);

        if (gotFrame) {
            // Fire DMA and return immediately — CPU does no pixel work at all.
            // frameFb[dmsg.bufSet] must stay valid until dmaPending clears, which
            // the ping-pong buffer design guarantees (decoder won't reclaim it
            // until we signal completion after waitDMA).
            //
            // Claim the shared SPI bus for the WHOLE push, not just the
            // startWrite: the DMA runs on past this function, and touchTask
            // must not be able to slip a 1 MHz read in between. portMAX_DELAY
            // is safe — the only other holder is touchTask, which holds it for
            // one bounded 456 us read and is priority-boosted to get out of the
            // way the instant we ask for it.
            busLock();
            lcd.startWrite();
            lcd.setAddrWindow(0, 0, SCREEN_W, SCREEN_H);
            lcd.pushPixelsDMA(frameFb[dmsg.bufSet], SCREEN_W * SCREEN_H);
            dmaPending = true;
            continue;  // loop back immediately; don't touch bus until DMA is done
        }

        // ── No frame and no DMA in flight — handle streaming overlay ──────────
        if (!g_streaming) {
            overlayVisible = false;
            esp_task_wdt_reset();
            continue;
        }

        uint32_t now = millis();
        if ((now - g_lastPktMs) < PKT_TIMEOUT_MS) {
            overlayVisible = false;
            esp_task_wdt_reset();
            continue;
        }

        bool wifiDisconnected = !g_wifiOk;
        const char* line1  = wifiDisconnected ? "WIFI DISCONNECTED" : "WAITING FOR VIDEO";
        const char* line2  = wifiDisconnected ? "TRY CONNECT BACK"  : "";
        uint32_t bgColor   = wifiDisconnected ? 0x2000 : 0x0841;
        uint32_t textColor = wifiDisconnected ? TFT_WHITE : TFT_YELLOW;

        if (!overlayVisible || (now - lastOverlayMs) >= OVERLAY_FLASH_MS) {
            lastOverlayMs  = now;
            overlayVisible = true;

            const int OX = 4, OY = 4, OW = 236, OH = 38;

            // Bracket all overlay draw calls in one bus transaction to avoid
            // repeated SPI lock/unlock overhead per primitive.
            busLock();
            lcd.startWrite();
            lcd.fillRect(OX, OY, OW, OH, bgColor);
            lcd.setTextFont(2);
            lcd.setTextSize(1);
            lcd.setTextColor(textColor, bgColor);
            lcd.drawString(line1, OX + 4, OY + 3);
            if (line2[0]) lcd.drawString(line2, OX + 4, OY + 20);
            lcd.endWrite();
            busUnlock();
        }

        esp_task_wdt_reset();
    }
}