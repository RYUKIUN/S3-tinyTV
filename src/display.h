#pragma once

#include "shared.h"
#include <LovyanGFX.hpp>

class LGFX : public lgfx::LGFX_Device {
    lgfx::Bus_SPI        _bus;
    lgfx::Panel_ILI9341  _panel;
public:
    LGFX() {
        {
            auto cfg = _bus.config();

            // ── SPI host ──────────────────────────────────────────────────────
            // USE_HSPI_PORT=1 in the original build flags means SPI2 host.
            // LovyanGFX uses spi_host_device_t: SPI2_HOST = 1, SPI3_HOST = 2.
            cfg.spi_host   = SPI2_HOST;   // SPI2 (HSPI)
            cfg.freq_write = 75000000;    // 60 MHz write clock

            // ── Pins (from build_flags) ───────────────────────────────────────
            cfg.pin_sclk = 12;   // TFT_SCLK
            cfg.pin_mosi = 13;   // TFT_MOSI
            cfg.pin_miso = -1;   // not connected
            cfg.pin_dc   =  4;   // TFT_DC  (Data/Command)

            cfg.spi_3wire  = false;  // 4-wire SPI (MOSI + DC line)
            cfg.use_lock   = true;   // safe for multi-device SPI bus

            _bus.config(cfg);
            _panel.setBus(&_bus);
        }
        {
            auto cfg = _panel.config();

            cfg.pin_cs   = 10;   // TFT_CS
            cfg.pin_rst  =  5;   // TFT_RST
            cfg.pin_busy = -1;

            cfg.panel_width  = 240;
            cfg.panel_height = 320;
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