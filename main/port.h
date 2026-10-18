#pragma once
/*
 * port.h — the entire "porting layer" this rebase needed.
 * ─────────────────────────────────────────────────────────
 * The old codebase used millis()/micros()/delay()/constrain() everywhere
 * because that's what Arduino gives you for free. None of that is Arduino
 * *framework* behavior worth dragging a whole extra build in for — it's
 * three one-line wrappers around esp_timer and vTaskDelay, and a clamp
 * macro. Defining them here once means the rest of the project (main,
 * network, touch, display) didn't need to be rewritten statement-by-
 * statement, while the project is still 100% pure ESP-IDF underneath:
 * no arduino-esp32 component is compiled or linked anywhere in this tree.
 */

#include <stdint.h>
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static inline uint32_t millis() {
    return (uint32_t)(esp_timer_get_time() / 1000ULL);
}

static inline uint32_t micros() {
    return (uint32_t)esp_timer_get_time();
}

static inline void delay(uint32_t ms) {
    vTaskDelay(pdMS_TO_TICKS(ms));
}

// Arduino's constrain() is a type-loose macro, not a function — keep it a
// macro here too so `constrain(int32_t, 0, SCREEN_W-1)` style call sites
// (mismatched int/int32_t literals) keep compiling exactly as before.
#ifndef constrain
#define constrain(amt, low, high) \
    ((amt) < (low) ? (low) : ((amt) > (high) ? (high) : (amt)))
#endif
