#pragma once

#include "shared.h"
#include <LovyanGFX.hpp>

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

extern LGFX lcd;

void initDisplay();
void drawBootHeader();
void statusLine(uint8_t row, const char* label, const char* value, uint32_t col = TFT_WHITE);
void displayTask(void*);
