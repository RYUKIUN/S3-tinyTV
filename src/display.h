#pragma once

#include "nexus.h"
#include <LovyanGFX.hpp>

// All pin numbers and SPI config below come from nexus.h (Zone 1) — there's
// nothing hardcoded here. If your panel is wired differently, edit the
// LCD_PIN_* / LCD_* defines in nexus.h, not this file.
class LGFX : public lgfx::LGFX_Device {
    lgfx::Bus_SPI        _bus;
    lgfx::Panel_ILI9341  _panel;
public:
    LGFX() {
        {
            auto cfg = _bus.config();

            cfg.spi_host   = LCD_SPI_HOST;
            cfg.freq_write = LCD_WRITE_HZ;

            cfg.pin_sclk = LCD_PIN_SCLK;
            cfg.pin_mosi = LCD_PIN_MOSI;
            cfg.pin_miso = LCD_PIN_MISO;
            cfg.pin_dc   = LCD_PIN_DC;

            cfg.spi_3wire  = false;  // 4-wire SPI (MOSI + DC line)
            cfg.use_lock   = true;   // safe for multi-device SPI bus

            _bus.config(cfg);
            _panel.setBus(&_bus);
        }
        {
            auto cfg = _panel.config();

            cfg.pin_cs   = LCD_PIN_CS;
            cfg.pin_rst  = LCD_PIN_RST;
            cfg.pin_busy = LCD_PIN_BUSY;

            cfg.panel_width  = LCD_PANEL_W;
            cfg.panel_height = LCD_PANEL_H;
            cfg.offset_x         = 0;
            cfg.offset_y         = 0;
            cfg.offset_rotation  = 0;
            cfg.dummy_read_pixel = 8;

            cfg.readable    = false;
            cfg.invert      = false;
            cfg.rgb_order   = false;
            cfg.dlen_16bit  = false;
            cfg.bus_shared  = true;   // CS must be driven; touch shares bus

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