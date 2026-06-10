#include "display.h"

LGFX lcd;

void initDisplay() {
    lcd.init();
    lcd.setRotation(1);
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
    DisplayMsg dmsg;
    bool overlayVisible = false;
    uint32_t lastOverlayMs = 0;

    while (true) {
        // ── Offline-mode shutdown ─────────────────────────────────────────────
        // offlinePlayerTask calls lcd.pushImage() directly from Core 1.
        // displayTask must exit before that starts to avoid concurrent LCD access.
        // enterOfflineMode() waits 500 ms after setting this flag, giving us
        // plenty of time to reach this check and self-delete.
        if (g_offlineMode) {
            Serial.println("[DISP] Offline mode — displayTask exiting");
            vTaskDelete(NULL);
            return;
        }

        bool gotFrame = (xQueueReceive(displayQueue, &dmsg, pdMS_TO_TICKS(100)) == pdTRUE);

        if (gotFrame) {
            lcd.pushImage(0, 0, SCREEN_W, SCREEN_H, frameFb[dmsg.bufSet]);
            g_presentedFrames++;
            overlayVisible = false;
            vTaskDelay(pdMS_TO_TICKS(1));
        }

        if (!g_streaming) {
            overlayVisible = false;
            continue;
        }

        uint32_t now = millis();
        bool timedOut = ((now - g_lastPktMs) >= PKT_TIMEOUT_MS);
        if (!timedOut) {
            overlayVisible = false;
            continue;
        }

        bool wifiDisconnected = !g_wifiOk;
        const char* line1 = wifiDisconnected ? "WIFI DISCONNECTED" : "WAITING FOR VIDEO";
        const char* line2 = wifiDisconnected ? "TRY CONNECT BACK" : "";
        uint32_t bgColor = wifiDisconnected ? 0x2000 : 0x0841;
        uint32_t textColor = wifiDisconnected ? TFT_WHITE : TFT_YELLOW;

        if (!overlayVisible || (now - lastOverlayMs) >= OVERLAY_FLASH_MS) {
            lastOverlayMs = now;
            overlayVisible = true;

            const int OX = 4;
            const int OY = 4;
            const int OW = 236;
            const int OH = 38;

            lcd.fillRect(OX, OY, OW, OH, bgColor);
            lcd.setTextFont(2);
            lcd.setTextSize(1);
            lcd.setTextColor(textColor, bgColor);
            lcd.drawString(line1, OX + 4, OY + 3);
            if (line2[0]) {
                lcd.drawString(line2, OX + 4, OY + 20);
            }
        }
    }
}