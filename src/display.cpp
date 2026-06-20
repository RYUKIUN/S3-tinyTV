#include "display.h"
#include "esp_task_wdt.h"

LGFX lcd;

void initDisplay() {
    lcd.init();
    lcd.setRotation(3);
    lcd.setColorDepth(16);
    lcd.setTextFont(2);
    lcd.setTextSize(1);
}

void statusLine(uint8_t row, const char* label, const char* value, uint32_t col) {
    int y = 58 + row * 22;
    lcd.fillRect(0, y, SCREEN_W, 22, TFT_BLACK);
    lcd.setTextColor(0x7BEF, TFT_BLACK);
    lcd.drawString(label, 8, y + 3);
    lcd.setTextColor(col, TFT_BLACK);
    lcd.drawString(value, 138, y + 3);
}

void drawBootHeader() {
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
}

void displayTask(void*) {
    esp_task_wdt_add(NULL);  // register this task as its own TWDT subscriber

    DisplayMsg dmsg;
    bool dmaPending      = false;  // true while a pushPixelsDMA transfer is in flight
    bool overlayVisible  = false;
    uint32_t lastOverlayMs  = 0;
    uint32_t dmaStartUs     = 0;
    uint32_t dmaPushAcc     = 0;
    uint32_t dmaPushCount   = 0;

    while (true) {
        // ── Offline-mode shutdown ─────────────────────────────────────────────
        // offlinePlayerTask calls lcd.pushImage() directly from Core 1.
        // displayTask must exit before that starts to avoid concurrent LCD access.
        // enterOfflineMode() waits 500 ms after setting this flag, giving us
        // plenty of time to reach this check and self-delete.
        if (g_offlineMode) {
            if (dmaPending) {
                lcd.waitDMA();
                lcd.endWrite();
                dmaPending = false;
            }
            Serial.println("[DISP] Offline mode — displayTask exiting");
            esp_task_wdt_delete(NULL);
            vTaskDelete(NULL);
            return;
        }

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
            dmaPending = false;
            g_presentedFrames++;
            overlayVisible = false;

            dmaPushAcc += (micros() - dmaStartUs);
            dmaPushCount++;
            if (dmaPushCount >= 16) {
                g_avgDmaPushUs = dmaPushAcc / dmaPushCount;
                dmaPushAcc = 0; dmaPushCount = 0;
            }

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
            lcd.startWrite();
            lcd.setAddrWindow(0, 0, SCREEN_W, SCREEN_H);
            dmaStartUs = micros();
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
            lcd.startWrite();
            lcd.fillRect(OX, OY, OW, OH, bgColor);
            lcd.setTextFont(2);
            lcd.setTextSize(1);
            lcd.setTextColor(textColor, bgColor);
            lcd.drawString(line1, OX + 4, OY + 3);
            if (line2[0]) lcd.drawString(line2, OX + 4, OY + 20);
            lcd.endWrite();
        }

        esp_task_wdt_reset();
    }
}