#include "display.h"
#include "esp_task_wdt.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ops.h"
#include "esp_lcd_io_spi.h"
#include "esp_lcd_ili9341.h"
#if DISPLAY_SELECT == DISPLAY_TFT24_I80
#include "esp_lcd_io_i80.h"
#endif

static const char* TAG = "display";

lgfx::LGFX_Sprite lcd;                 // software canvas (no bus attached)

static esp_lcd_panel_io_handle_t s_io    = nullptr;
static esp_lcd_panel_handle_t    s_panel = nullptr;

static uint8_t*  s_canvasBuf = nullptr;      // SCREEN_W*SCREEN_H*2, PSRAM, 64 B aligned
static lgfx::LGFX_Sprite s_ovl;              // streaming-overlay sprite
static uint8_t*  s_ovlBuf    = nullptr;      // internal, DMA-capable
static const int OVL_X = 4, OVL_Y = 4, OVL_W = 236, OVL_H = 38;

static SemaphoreHandle_t s_doneSem = nullptr;   // given by the ISR when the LAST strip finishes
static volatile int32_t  s_stripsLeft = 0;
static SemaphoreHandle_t s_uiMutex = nullptr;   // recursive; guards the canvas sprite

// Bus ownership helpers. Null-safe because drawBootHeader()/statusLine() run
// during early app_main(), potentially before the mutex object exists — and a
// single-threaded boot needs no arbitration anyway.
static inline void busLock()   { if (g_spiMutex) xSemaphoreTake(g_spiMutex, portMAX_DELAY); }
static inline void busUnlock() { if (g_spiMutex) xSemaphoreGive(g_spiMutex); }

void displayUiLock()   { if (s_uiMutex) xSemaphoreTakeRecursive(s_uiMutex, portMAX_DELAY); }
void displayUiUnlock() { if (s_uiMutex) xSemaphoreGiveRecursive(s_uiMutex); }

// ── Completion callback (ISR context, IRAM-safe: SPI_MASTER_ISR_IN_IRAM=y) ──
// Runs once per finished colour transaction (= one strip). The last one wakes
// the task that is sleeping in displayWaitDone().
static bool IRAM_ATTR onColorDone(esp_lcd_panel_io_handle_t, esp_lcd_panel_io_event_data_t*, void*) {
    BaseType_t hpw = pdFALSE;
    if (__atomic_sub_fetch(&s_stripsLeft, 1, __ATOMIC_ACQ_REL) == 0)
        xSemaphoreGiveFromISR(s_doneSem, &hpw);
    return hpw == pdTRUE;
}

static bool waitDone(uint32_t timeoutMs) {
    return xSemaphoreTake(s_doneSem, pdMS_TO_TICKS(timeoutMs)) == pdTRUE;
}

// Queue a w x h rectangle (src is tightly packed, big-endian RGB565) as a
// chain of strips and return WITHOUT waiting. Caller must hold the bus mutex
// and then wait with waitDone() before releasing it / reusing `src`.
// Window is set once; the first strip uses RAMWR (0x2C), the rest RAMWRC (0x3C)
// so the controller keeps filling the same window with no per-strip setup.
static bool pushRectAsync(const uint8_t* src, int x0, int y0, int w, int h) {
    if (!s_io || w <= 0 || h <= 0) return false;

    const uint8_t ca[4] = { (uint8_t)(x0 >> 8), (uint8_t)x0,
                            (uint8_t)((x0 + w - 1) >> 8), (uint8_t)(x0 + w - 1) };
    const uint8_t pa[4] = { (uint8_t)(y0 >> 8), (uint8_t)y0,
                            (uint8_t)((y0 + h - 1) >> 8), (uint8_t)(y0 + h - 1) };
    if (esp_lcd_panel_io_tx_param(s_io, 0x2A, ca, 4) != ESP_OK) return false;
    if (esp_lcd_panel_io_tx_param(s_io, 0x2B, pa, 4) != ESP_OK) return false;

    const int strips = (h + LCD_STRIP_ROWS - 1) / LCD_STRIP_ROWS;
    xSemaphoreTake(s_doneSem, 0);                       // drop any stale token
    __atomic_store_n(&s_stripsLeft, strips, __ATOMIC_RELEASE);

    const size_t rowBytes = (size_t)w * 2;
    int row = 0;
    for (int s = 0; s < strips; s++) {
        int n = (h - row < LCD_STRIP_ROWS) ? (h - row) : LCD_STRIP_ROWS;
        esp_err_t e = esp_lcd_panel_io_tx_color(s_io, s == 0 ? 0x2C : 0x3C,
                                                src + (size_t)row * rowBytes, (size_t)n * rowBytes);
        if (e != ESP_OK) {
            ESP_LOGE(TAG, "tx_color strip %d failed: %s", s, esp_err_to_name(e));
            // Strips [s..] will never complete; settle the counter for them.
            int left = __atomic_sub_fetch(&s_stripsLeft, strips - s, __ATOMIC_ACQ_REL);
            if (s > 0 && left > 0) waitDone(DISPLAY_DMA_TIMEOUT_MS);   // let queued ones drain
            return false;
        }
        row += n;
    }
    return true;
}

static bool pushRectBlocking(const uint8_t* src, int x0, int y0, int w, int h) {
    busLock();
    bool ok = pushRectAsync(src, x0, y0, w, h);
    if (ok && !waitDone(DISPLAY_DMA_TIMEOUT_MS)) {
        ESP_LOGE(TAG, "DMA completion timeout (UI push)");
        ok = false;
    }
    busUnlock();
    return ok;
}

void displayFlushRows(int y0, int rows) {
    if (!s_canvasBuf) return;
    if (y0 < 0) { rows += y0; y0 = 0; }
    if (y0 + rows > SCREEN_H) rows = SCREEN_H - y0;
    if (rows <= 0) return;
    pushRectBlocking(s_canvasBuf + (size_t)y0 * SCREEN_W * 2, 0, y0, SCREEN_W, rows);
}

void displayFlush() { displayFlushRows(0, SCREEN_H); }

// Drive strength is a pad (IO_MUX) property, so it survives whatever function
// the pin is routed to — but the IDF drivers may reconfigure pads during init,
// so this is applied AFTER initDisplay().
void displayTuneBus() {
#if DISPLAY_SELECT == DISPLAY_TFT24_I80
    const int pins[] = { LCD_I80_PIN_WR,
        LCD_I80_PIN_D0, LCD_I80_PIN_D1, LCD_I80_PIN_D2, LCD_I80_PIN_D3,
        LCD_I80_PIN_D4, LCD_I80_PIN_D5, LCD_I80_PIN_D6, LCD_I80_PIN_D7 };
    for (int p : pins)
        if (p >= 0) gpio_set_drive_capability((gpio_num_t)p, (gpio_drive_cap_t)LCD_I80_DRIVE_CAP);
    ESP_LOGI(TAG, "i80 8-bit %u Hz, drive cap %d", (unsigned)LCD_I80_WRITE_HZ, LCD_I80_DRIVE_CAP);
#else
    gpio_set_drive_capability((gpio_num_t)LCD_PIN_SCLK, (gpio_drive_cap_t)LCD_DRIVE_CAP);
    gpio_set_drive_capability((gpio_num_t)LCD_PIN_MOSI, (gpio_drive_cap_t)LCD_DRIVE_CAP);
    ESP_LOGI(TAG, "SPI %u Hz, drive cap %d", (unsigned)LCD_WRITE_HZ, LCD_DRIVE_CAP);
#endif
}

static bool createPanelIo() {
#if DISPLAY_SELECT == DISPLAY_TFT24_I80
    // UNTESTED path (no i80 hardware was available) — same IO abstraction as SPI.
    // esp_lcd's i80 bus has no RD pin; hold it high ourselves (panel is write-only).
    if (LCD_I80_PIN_RD >= 0) {
        gpio_config_t rd = {};
        rd.pin_bit_mask = 1ULL << LCD_I80_PIN_RD;
        rd.mode = GPIO_MODE_OUTPUT;
        gpio_config(&rd);
        gpio_set_level((gpio_num_t)LCD_I80_PIN_RD, 1);
    }
    esp_lcd_i80_bus_handle_t bus = nullptr;
    esp_lcd_i80_bus_config_t bc = {};
    bc.dc_gpio_num = (gpio_num_t)LCD_I80_PIN_RS;
    bc.wr_gpio_num = (gpio_num_t)LCD_I80_PIN_WR;
    bc.clk_src     = LCD_CLK_SRC_DEFAULT;
    const int d[8] = { LCD_I80_PIN_D0, LCD_I80_PIN_D1, LCD_I80_PIN_D2, LCD_I80_PIN_D3,
                       LCD_I80_PIN_D4, LCD_I80_PIN_D5, LCD_I80_PIN_D6, LCD_I80_PIN_D7 };
    for (int i = 0; i < 8; i++) bc.data_gpio_nums[i] = (gpio_num_t)d[i];
    bc.bus_width           = 8;
    bc.max_transfer_bytes  = LCD_STRIP_ROWS * SCREEN_W * 2;
    bc.psram_trans_align   = LCD_FB_ALIGN;
    bc.sram_trans_align    = 4;
    if (esp_lcd_new_i80_bus(&bc, &bus) != ESP_OK) { ESP_LOGE(TAG, "i80 bus init failed"); return false; }

    esp_lcd_panel_io_i80_config_t ic = {};
    ic.cs_gpio_num        = (gpio_num_t)LCD_I80_PIN_CS;
    ic.pclk_hz            = LCD_I80_WRITE_HZ;
    ic.trans_queue_depth  = LCD_TRANS_QUEUE;
    ic.dc_levels.dc_idle_level  = 0;
    ic.dc_levels.dc_cmd_level   = 0;
    ic.dc_levels.dc_dummy_level = 0;
    ic.dc_levels.dc_data_level  = 1;
    ic.lcd_cmd_bits       = 8;
    ic.lcd_param_bits     = 8;
    ic.on_color_trans_done = onColorDone;
    return esp_lcd_new_panel_io_i80(bus, &ic, &s_io) == ESP_OK;
#else
    spi_bus_config_t bc = {};
    bc.sclk_io_num     = LCD_PIN_SCLK;
    bc.mosi_io_num     = LCD_PIN_MOSI;
    // The panel never reads, but the XPT2046 does — MISO must be on the bus.
    bc.miso_io_num     = TOUCH_FEATURE_ENABLED ? TOUCH_PIN_MISO : -1;
    bc.quadwp_io_num   = -1;
    bc.quadhd_io_num   = -1;
    bc.max_transfer_sz = LCD_STRIP_ROWS * SCREEN_W * 2;
    esp_err_t e = spi_bus_initialize(LCD_SPI_HOST, &bc, SPI_DMA_CH_AUTO);
    if (e != ESP_OK) { ESP_LOGE(TAG, "spi_bus_initialize: %s", esp_err_to_name(e)); return false; }

    uint32_t hz = LCD_WRITE_HZ;
    for (int attempt = 0; attempt < 2; attempt++, hz = 40000000) {
        esp_lcd_panel_io_spi_config_t ic = {};
        ic.cs_gpio_num       = (gpio_num_t)LCD_PIN_CS;
        ic.dc_gpio_num       = (gpio_num_t)LCD_PIN_DC;
        ic.spi_mode          = 0;
        ic.pclk_hz           = hz;
        ic.trans_queue_depth = LCD_TRANS_QUEUE;
        ic.lcd_cmd_bits      = 8;
        ic.lcd_param_bits    = 8;
        ic.on_color_trans_done = onColorDone;
        e = esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_SPI_HOST, &ic, &s_io);
        if (e == ESP_OK) { ESP_LOGI(TAG, "panel IO up at %u Hz", (unsigned)hz); return true; }
        ESP_LOGW(TAG, "panel IO at %u Hz failed (%s)%s", (unsigned)hz, esp_err_to_name(e),
                 attempt == 0 && hz != 40000000 ? " — retrying at 40 MHz" : "");
        if (hz == 40000000) break;
    }
    return false;
#endif
}

bool initDisplay() {
    s_doneSem = xSemaphoreCreateBinary();
    s_uiMutex = xSemaphoreCreateRecursiveMutex();
    if (!s_doneSem || !s_uiMutex) return false;

    if (!createPanelIo()) return false;

    esp_lcd_panel_dev_config_t pc = {};
#if DISPLAY_SELECT == DISPLAY_TFT24_I80
    pc.reset_gpio_num = (gpio_num_t)LCD_I80_PIN_RST;
#else
    pc.reset_gpio_num = (gpio_num_t)LCD_PIN_RST;
#endif
    pc.rgb_ele_order  = LCD_RGB_ELEMENT_ORDER_BGR;
    pc.bits_per_pixel = 16;
    if (esp_lcd_new_panel_ili9341(s_io, &pc, &s_panel) != ESP_OK) { ESP_LOGE(TAG, "ili9341 panel create failed"); return false; }
    esp_lcd_panel_reset(s_panel);
    esp_lcd_panel_init(s_panel);

    // Orientation/colour order is set explicitly (see LCD_MADCTL in nexus.h) so
    // it matches the old LovyanGFX rotation 3, independent of component defaults.
    uint8_t mad = LCD_MADCTL;
    esp_lcd_panel_io_tx_param(s_io, 0x36, &mad, 1);
    esp_lcd_panel_disp_on_off(s_panel, true);

    // Software canvas for UI text, backed by an aligned PSRAM buffer so the
    // driver can DMA straight out of it.
    const size_t canvasBytes = (size_t)SCREEN_W * SCREEN_H * 2;
    s_canvasBuf = (uint8_t*)heap_caps_aligned_alloc(LCD_FB_ALIGN, canvasBytes, MALLOC_CAP_SPIRAM);
    s_ovlBuf    = (uint8_t*)heap_caps_aligned_alloc(4, (size_t)OVL_W * OVL_H * 2,
                                                    MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    if (!s_canvasBuf || !s_ovlBuf) { ESP_LOGE(TAG, "UI buffer allocation failed"); return false; }
    memset(s_canvasBuf, 0, canvasBytes);

    lcd.setBuffer(s_canvasBuf, SCREEN_W, SCREEN_H, lgfx::rgb565_2Byte);
    lcd.fillScreen(TFT_BLACK);
    lcd.setTextFont(2);
    lcd.setTextSize(1);
    s_ovl.setBuffer(s_ovlBuf, OVL_W, OVL_H, lgfx::rgb565_2Byte);

    displayFlush();    // clear the panel (GRAM is undefined after reset)
    return true;
}

void statusLine(uint8_t row, const char* label, const char* value, uint32_t col) {
    int y = 58 + row * 22;
    displayUiLock();
    lcd.fillRect(0, y, SCREEN_W, 22, TFT_BLACK);
    lcd.setTextColor(0x7BEF, TFT_BLACK);
    lcd.drawString(label, 8, y + 3);
    lcd.setTextColor(col, TFT_BLACK);
    lcd.drawString(value, 138, y + 3);
    displayFlushRows(y, 22);
    displayUiUnlock();
}

void drawBootHeader() {
    displayUiLock();
    lcd.fillScreen(TFT_BLACK);
    lcd.setTextFont(2);
    lcd.setTextSize(1);
    lcd.fillRect(0, 0, SCREEN_W, 54, 0x1082);
    lcd.setTextColor(TFT_CYAN, 0x1082);
    lcd.setTextSize(2);
    lcd.drawString("ESP32-S3 STREAM", 8, 6);
    lcd.setTextSize(1);
    lcd.setTextColor(0x7BEF, 0x1082);
    lcd.drawString(LCD_MODEL_NAME "  ping-pong", 8, 34);
    lcd.drawFastHLine(0, 54, SCREEN_W, TFT_DARKGREY);
    displayFlush();
    displayUiUnlock();
}

void displayTask(void*) {
    esp_task_wdt_add(NULL);  // register this task as its own TWDT subscriber

    DisplayMsg dmsg;
    bool overlayVisible = false;
    uint32_t lastOverlayMs = 0;
    while (true) {
        // 8 ms timeout keeps the overlay/watchdog logic below responsive while idle.
        bool gotFrame = (xQueueReceive(displayQueue, &dmsg, pdMS_TO_TICKS(8)) == pdTRUE);

        if (gotFrame) {
            // Hold the shared bus for the WHOLE push: from the first strip being
            // queued until the last-strip callback fires. frameFb[bufSet] must
            // stay untouched until then, which the ping-pong design guarantees
            // (the decoder only gets the buffer back via framebufferFreeQueue).
            busLock();
            bool queued = pushRectAsync((const uint8_t*)frameFb[dmsg.bufSet], 0, 0, SCREEN_W, SCREEN_H);
            if (queued) {
                // SLEEP until the ISR reports the last strip done — no polling.
                uint32_t waited = 0;
                while (!waitDone(10)) {
                    esp_task_wdt_reset();
                    waited += 10;
                    if (waited >= DISPLAY_DMA_TIMEOUT_MS) {
                        ESP_LOGE(TAG, "frame DMA completion timeout");
                        queued = false;
                        break;
                    }
                }
            }
            busUnlock();

            if (dmsg.bufSet >= g_numDisplayBufs ||
                xQueueSend(framebufferFreeQueue, &dmsg.bufSet, 0) != pdTRUE) {
                ESP_LOGE("pipeline", "display framebuffer %u could not be returned", dmsg.bufSet);
            }
            if (queued) g_presentedFrames = g_presentedFrames + 1;
            overlayVisible = false;
            esp_task_wdt_reset();
            continue;
        }

        // ── No frame pending — handle streaming overlay ──────────────────────
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

            // Render into the small overlay sprite (only this task touches it),
            // then send just that rectangle over the last video frame.
            s_ovl.fillScreen(bgColor);
            s_ovl.setTextFont(2);
            s_ovl.setTextSize(1);
            s_ovl.setTextColor(textColor, bgColor);
            s_ovl.drawString(line1, 4, 3);
            if (line2[0]) s_ovl.drawString(line2, 4, 20);
            pushRectBlocking(s_ovlBuf, OVL_X, OVL_Y, OVL_W, OVL_H);
        }

        esp_task_wdt_reset();
    }
}
