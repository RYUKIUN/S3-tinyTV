#include "display.h"

class LGFX : public lgfx::LGFX_Device {
    lgfx::Bus_Parallel8  _bus;
    lgfx::Panel_ILI9341  _panel;
public:
    LGFX() {
        {
            auto cfg = _bus.config();
            cfg.freq_write = 30000000;
            cfg.pin_wr = 1; cfg.pin_rd = 40; cfg.pin_rs = 2;
            cfg.pin_d0 = 5; cfg.pin_d1 = 4;  cfg.pin_d2 = 10;
            cfg.pin_d3 = 9; cfg.pin_d4 = 3;  cfg.pin_d5 = 8;
            cfg.pin_d6 = 7; cfg.pin_d7 = 6;
            _bus.config(cfg);
            _panel.setBus(&_bus);
        }
        {
            auto cfg = _panel.config();
            cfg.pin_cs = 41; cfg.pin_rst = 39; cfg.pin_busy = -1;
            cfg.panel_width = 240; cfg.panel_height = 320;
            cfg.offset_x = 0; cfg.offset_y = 0; cfg.offset_rotation = 0;
            cfg.dummy_read_pixel = 8;
            cfg.readable = false; cfg.invert = false;
            cfg.rgb_order = false; cfg.dlen_16bit = false; cfg.bus_shared = false;
            _panel.config(cfg);
        }
        setPanel(&_panel);
    }
};

static LGFX lcd;

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
        bool gotFrame = (xQueueReceive(displayQueue, &dmsg, pdMS_TO_TICKS(100)) == pdTRUE);

        if (gotFrame) {
            lcd.pushImage(0, 0, SCREEN_W, SCREEN_H, frameFb[dmsg.bufSet]);
            g_presentedFrames++;
            overlayVisible = false;
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
