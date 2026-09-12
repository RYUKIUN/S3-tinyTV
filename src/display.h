#pragma once

#include "nexus.h"
#include <LovyanGFX.hpp>

// All pin numbers and SPI config below come from nexus.h (Zone 1) — there's
// nothing hardcoded here. If your panel is wired differently, edit the
// LCD_PIN_* / LCD_* defines in nexus.h, not this file.
class LGFX : public lgfx::LGFX_Device {
    lgfx::Bus_SPI          _bus;
    lgfx::Panel_ILI9341    _panel;
#if TOUCH_FEATURE_ENABLED
    lgfx::Touch_XPT2046    _touch;
#endif
public:
    LGFX() {
        {
            auto cfg = _bus.config();

            cfg.spi_host   = LCD_SPI_HOST;
            cfg.freq_write = LCD_WRITE_HZ;

            cfg.pin_sclk = LCD_PIN_SCLK;
            cfg.pin_mosi = LCD_PIN_MOSI;
            // The PANEL is write-only (LCD_PIN_MISO is -1 and readable=false),
            // but the BUS now carries the XPT2046's read line, so MISO has to
            // be part of the bus config from the very first init.
            //
            // This is load-bearing, not tidiness. LovyanGFX's spi::init guards
            // spi_bus_initialize() behind "have I already made a device handle
            // for this host", so the second call — the one Touch_XPT2046 makes
            // during initTouch() — skips it entirely and cannot retrofit MISO
            // into the IDF bus configuration. Declaring it here means the one
            // spi_bus_initialize() that does run already has it.
            cfg.pin_miso = TOUCH_FEATURE_ENABLED ? TOUCH_PIN_MISO : -1;
            cfg.pin_dc   = LCD_PIN_DC;

            cfg.spi_3wire  = false;  // 4-wire SPI (MOSI + DC line)
            cfg.use_lock   = true;   // safe for multi-device SPI bus

            _bus.config(cfg);
            _panel.setBus(&_bus);
        }
    #if TOUCH_FEATURE_ENABLED
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
        {
            // XPT2046 on the same SPI host as the panel. LovyanGFX's touch
            // init calls lgfx::spi::init() on that host with a MISO pin, which
            // is what actually brings the read line onto the shared bus (the
            // panel's own config leaves miso at -1 because it never reads).
            //
            // The driver median-filters 7 hardware samples per read internally,
            // so there's no filtering left for us to do.
            //
            // pin_int is gated on TOUCH_USE_IRQ and must stay -1 while PENIRQ
            // is unwired. getTouchRaw() checks that level FIRST and bails if
            // it's high: with the line wired that's the whole win (an untouched
            // poll becomes one GPIO read and zero SPI traffic), but with it
            // floating it would make every read report "not touched" forever.
            // Touch detection itself does not need PENIRQ — the XPT2046
            // measures pressure, and the driver requires a valid Z reading
            // before it reports a point.
            auto cfg = _touch.config();

            cfg.spi_host = LCD_SPI_HOST;      // share the display's bus
            cfg.freq     = TOUCH_SPI_HZ;
            cfg.pin_sclk = LCD_PIN_SCLK;
            cfg.pin_mosi = LCD_PIN_MOSI;
            cfg.pin_miso = TOUCH_PIN_MISO;
            cfg.pin_cs   = TOUCH_PIN_CS;
            cfg.pin_int  = TOUCH_USE_IRQ ? TOUCH_PIN_IRQ : -1;
            cfg.bus_shared = true;

            // Raw-ADC bounds are only the pre-calibration fallback; once a
            // calibration is loaded, setTouchCalibrate()'s affine transform
            // supersedes them entirely.
            cfg.x_min = 300;  cfg.x_max = 3900;
            cfg.y_min = 400;  cfg.y_max = 3900;
            cfg.offset_rotation = 0;

            _touch.config(cfg);
            _panel.setTouch(&_touch);
        }
#endif
        setPanel(&_panel);
    }
};

extern LGFX lcd;

void initDisplay();
void drawBootHeader();
void statusLine(uint8_t row, const char* label, const char* value, uint32_t col = TFT_WHITE);
void displayTask(void*);