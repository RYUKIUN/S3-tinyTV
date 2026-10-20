#pragma once

#include "nexus.h"
#include <LovyanGFX.hpp>

// All pin numbers and bus config below come from nexus.h (Zone 1) — there's
// nothing hardcoded here. Choose the display with DISPLAY_SELECT in nexus.h;
// edit LCD_PIN_* (SPI) or LCD_I80_PIN_* (i80 shield) there, not in this file.
class LGFX : public lgfx::LGFX_Device {
#if DISPLAY_SELECT == DISPLAY_TFT24_I80
    lgfx::Bus_Parallel8    _bus;     // 8-bit i80 (LCD_CAM on ESP32-S3)
#else
    lgfx::Bus_SPI          _bus;
#endif
    lgfx::Panel_ILI9341    _panel;
#if TOUCH_FEATURE_ENABLED
    lgfx::Touch_XPT2046    _touch;
#endif
public:
    LGFX() {
#if DISPLAY_SELECT == DISPLAY_TFT24_I80
        {
            auto cfg = _bus.config();

            cfg.freq_write = LCD_I80_WRITE_HZ;
            cfg.pin_wr = LCD_I80_PIN_WR;
            cfg.pin_rd = LCD_I80_PIN_RD;
            cfg.pin_rs = LCD_I80_PIN_RS;   // D/C
            cfg.pin_d0 = LCD_I80_PIN_D0;
            cfg.pin_d1 = LCD_I80_PIN_D1;
            cfg.pin_d2 = LCD_I80_PIN_D2;
            cfg.pin_d3 = LCD_I80_PIN_D3;
            cfg.pin_d4 = LCD_I80_PIN_D4;
            cfg.pin_d5 = LCD_I80_PIN_D5;
            cfg.pin_d6 = LCD_I80_PIN_D6;
            cfg.pin_d7 = LCD_I80_PIN_D7;

            _bus.config(cfg);
            _panel.setBus(&_bus);
        }
        {
            auto cfg = _panel.config();

            cfg.pin_cs   = LCD_I80_PIN_CS;
            cfg.pin_rst  = LCD_I80_PIN_RST;
            cfg.pin_busy = -1;

            cfg.panel_width  = LCD_PANEL_W;
            cfg.panel_height = LCD_PANEL_H;
            cfg.offset_x         = 0;
            cfg.offset_y         = 0;
            cfg.offset_rotation  = 0;
            cfg.dummy_read_pixel = 8;

            cfg.readable    = false;
            cfg.invert      = false;   // flip if colours look inverted
            cfg.rgb_order   = false;   // flip if red/blue are swapped
            cfg.dlen_16bit  = false;   // 8-bit bus
            cfg.bus_shared  = false;   // parallel bus is dedicated; no touch here

            _panel.config(cfg);
        }
#else
        {
            auto cfg = _bus.config();

            cfg.spi_host   = LCD_SPI_HOST;
            cfg.freq_write = LCD_WRITE_HZ;

            cfg.pin_sclk = LCD_PIN_SCLK;
            cfg.pin_mosi = LCD_PIN_MOSI;
            // The PANEL is write-only (LCD_PIN_MISO is -1 and readable=false),
            // but the BUS carries the XPT2046's read line when touch is on, so
            // MISO has to be part of the bus config from the very first init
            // (LovyanGFX's spi::init only runs spi_bus_initialize() once).
            cfg.pin_miso = TOUCH_FEATURE_ENABLED ? TOUCH_PIN_MISO : -1;
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
            cfg.bus_shared  = TOUCH_FEATURE_ENABLED ? true : false;  // CS must be driven when touch shares the bus

            _panel.config(cfg);
        }
    #if TOUCH_FEATURE_ENABLED
        {
            // XPT2046 on the same SPI host as the panel. LovyanGFX's touch
            // init calls lgfx::spi::init() on that host with a MISO pin, which
            // is what actually brings the read line onto the shared bus.
            //
            // pin_int is gated on TOUCH_USE_IRQ and must stay -1 while PENIRQ
            // is unwired (a floating line would make every read report
            // "not touched").
            auto cfg = _touch.config();

            cfg.spi_host = LCD_SPI_HOST;      // share the display's bus
            cfg.freq     = TOUCH_SPI_HZ;
            cfg.pin_sclk = LCD_PIN_SCLK;
            cfg.pin_mosi = LCD_PIN_MOSI;
            cfg.pin_miso = TOUCH_PIN_MISO;
            cfg.pin_cs   = TOUCH_PIN_CS;
            cfg.pin_int  = TOUCH_USE_IRQ ? TOUCH_PIN_IRQ : -1;
            cfg.bus_shared = true;

            // Raw-ADC bounds are only the pre-calibration fallback.
            cfg.x_min = 300;  cfg.x_max = 3900;
            cfg.y_min = 400;  cfg.y_max = 3900;
            cfg.offset_rotation = 0;

            _touch.config(cfg);
            _panel.setTouch(&_touch);
        }
    #endif
#endif
        setPanel(&_panel);
    }
};

extern LGFX lcd;

void initDisplay();
void displayTuneBus();      // pad drive strength; call after lcd.init() and touchBegin()
void drawBootHeader();
void statusLine(uint8_t row, const char* label, const char* value, uint32_t col = TFT_WHITE);
void displayTask(void*);