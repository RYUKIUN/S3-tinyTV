#include "network.h"
#include "display.h"
#include "wifi.h"
#include "nexus.h"
#include "esp_heap_caps.h"
#include "driver/temperature_sensor.h"
#include "esp_wifi.h"
#include "esp_log.h"

/*
 * REBASE NOTE: only two things changed here vs. the Arduino version —
 *  1. WiFi.status()/.RSSI()/.localIP().toString()/.begin()/.disconnect()
 *     became wifiReconnect() + wifiGetRSSI() + wifiGetIPString() (wifi.h),
 *     and g_wifiOk (set asynchronously by wifi.cpp's event handler) instead
 *     of polling WiFi.status() synchronously.
 *  2. Arduino's temperatureRead() became the native ESP-IDF temperature
 *     sensor driver, which needs a one-time install and enable.
 * The video transport reassembles complete JPEG frames from UDP chunks in
 * PSRAM; the stats packet and touch event protocol remain unchanged.
 */

static const char* TAG = "network";
static temperature_sensor_handle_t s_tempSensor = nullptr;
static bool s_tempSensorInitAttempted = false;
static bool s_tempSensorReady = false;

static float readChipTempC() {
    if (!s_tempSensorInitAttempted) {
        s_tempSensorInitAttempted = true;
        const temperature_sensor_config_t cfg = TEMPERATURE_SENSOR_CONFIG_DEFAULT(20, 100);
        esp_err_t err = temperature_sensor_install(&cfg, &s_tempSensor);
        if (err == ESP_OK) {
            err = temperature_sensor_enable(s_tempSensor);
        }
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "temperature sensor initialization failed: %s", esp_err_to_name(err));
            if (s_tempSensor != nullptr) {
                esp_err_t cleanupErr = temperature_sensor_uninstall(s_tempSensor);
                if (cleanupErr != ESP_OK) {
                    ESP_LOGW(TAG, "temperature sensor cleanup failed: %s", esp_err_to_name(cleanupErr));
                }
                s_tempSensor = nullptr;
            }
            return 0.0f;
        }
        s_tempSensorReady = true;
    }
    if (!s_tempSensorReady) return 0.0f;

    float tC = 0.0f;
    esp_err_t err = temperature_sensor_get_celsius(s_tempSensor, &tC);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "temperature sensor read failed: %s", esp_err_to_name(err));
        return 0.0f;
    }
    return tC;
}

static void resetFrameSlot(JpegFrameSlot& frame) {
    memset(frame.chunkReceived, 0, sizeof(frame.chunkReceived));
    frame.frameId = 0xFF;
    frame.totalChunks = 0;
    frame.chunksReceived = 0;
    frame.frameSize = 0;
    frame.firstChunkMs = 0;
    frame.receiving = false;
    frame.queued = false;
    frame.inUse = false;
}

// Drains touchTask's queue onto the wire. Called from networkTask's own loop so
// the UDP socket is only ever used by one task.
#if TOUCH_FEATURE_ENABLED
static void flushTouchEvents() {
    if (!g_remoteAddrValid || touchQueue == nullptr) return;

    TouchEvent ev;
    uint8_t    pkt[9];
    for (int i = 0; i < TOUCH_EVENT_QUEUE_LEN; i++) {
        if (xQueueReceive(touchQueue, &ev, 0) != pdTRUE) break;
        pkt[0] = 0xAA;
        pkt[1] = 0xDD;
        pkt[2] = ev.kind;
        pkt[3] = ev.touchId;
        pkt[4] = ev.seq;
        pkt[5] = (uint8_t)(ev.x >> 8);
        pkt[6] = (uint8_t)(ev.x & 0xFF);
        pkt[7] = (uint8_t)(ev.y >> 8);
        pkt[8] = (uint8_t)(ev.y & 0xFF);
        sendto(g_sock, pkt, sizeof(pkt), 0,
               (struct sockaddr*)&g_remoteAddr, sizeof(g_remoteAddr));
    }
}
#endif

void networkTask(void*) {
    g_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (g_sock < 0) {
        ESP_LOGE(TAG, "UDP socket creation failed");
        vTaskDelete(NULL);
        return;
    }

    int rcvbuf = 65536;
    setsockopt(g_sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

    struct sockaddr_in local = {};
    local.sin_family      = AF_INET;
    local.sin_port        = htons(UDP_PORT);
    local.sin_addr.s_addr = INADDR_ANY;
    if (bind(g_sock, (struct sockaddr*)&local, sizeof(local)) < 0) {
        ESP_LOGE(TAG, "UDP bind to port %d failed", UDP_PORT);
        close(g_sock);
        vTaskDelete(NULL);
        return;
    }
    fcntl(g_sock, F_SETFL, O_NONBLOCK);

    static uint8_t rxBuf[CHUNK_DATA_SIZE + 16];
    struct sockaddr_in sender;
    socklen_t slen = sizeof(sender);
    uint32_t lastPktMs = millis(), lastBeaconMs = 0, lastStatMs = 0;
    bool droppedFrameActive = false;
    uint8_t droppedFrameId = 0;

    auto returnJpegSlot = [&](uint8_t idx) {
        resetFrameSlot(jpegFrames[idx]);
        if (xQueueSend(jpegFreeQueue, &idx, 0) != pdTRUE) {
            ESP_LOGE(TAG, "JPEG slot %u could not be returned to free queue", idx);
        }
    };

    auto enqueueCompletedFrames = [&]() {
        for (uint8_t i = 0; i < g_numJpegSlots; ++i) {
            JpegFrameSlot& frame = jpegFrames[i];
            if (!frame.queued || frame.receiving) continue;

            DecodeMsg msg = { frame.frameId, i, frame.frameSize };
            if (xQueueSend(decodeQueue, &msg, 0) == pdTRUE) {
                frame.queued = false;
            }
        }
    };

    while (true) {
        enqueueCompletedFrames();

        uint32_t nowMs = millis();
        for (uint8_t i = 0; i < g_numJpegSlots; ++i) {
            JpegFrameSlot& frame = jpegFrames[i];
            if (!frame.receiving ||
                (nowMs - frame.firstChunkMs) <= FRAME_ASSEMBLY_TIMEOUT_MS) {
                continue;
            }

            g_networkDrops = g_networkDrops + 1;
            g_abortedFrames = g_abortedFrames + 1;
            if (droppedFrameActive && droppedFrameId == frame.frameId) {
                droppedFrameActive = false;
            }
            returnJpegSlot(i);
        }

        int n = recvfrom(g_sock, rxBuf, sizeof(rxBuf), 0,
                         (struct sockaddr*)&sender, &slen);

        if (n < 0) {
#if TOUCH_FEATURE_ENABLED
            flushTouchEvents();
#endif

            fd_set rfds; FD_ZERO(&rfds); FD_SET(g_sock, &rfds);
            struct timeval tv = { .tv_sec = 0, .tv_usec = 1000 };
            select(g_sock + 1, &rfds, NULL, NULL, &tv);

            if ((millis() - lastBeaconMs) > 2000 && (millis() - lastPktMs) > 2000) {
                struct sockaddr_in bc = {};
                bc.sin_family         = AF_INET;
                bc.sin_port           = htons(UDP_PORT);
                bc.sin_addr.s_addr    = htonl(INADDR_BROADCAST);
                int so = 1;
                setsockopt(g_sock, SOL_SOCKET, SO_BROADCAST, &so, sizeof(so));
                const char* b = "S3READY";
                sendto(g_sock, b, strlen(b), 0, (struct sockaddr*)&bc, sizeof(bc));
                lastBeaconMs = millis();
            }
            continue;
        }

        lastPktMs = millis();
        g_lastPktMs = lastPktMs;
        if (n < 4 || rxBuf[0] != 0xAA) { portYIELD(); continue; }
        memcpy(&g_remoteAddr, &sender, sizeof(sender));
        g_remoteAddrValid = true;

        if (rxBuf[1] == 0xCC) {
            if (n >= 4 && rxBuf[2] == 0x01) debugEnabled = (rxBuf[3] == 1);
#if TOUCH_FEATURE_ENABLED
            if (n >= 4 && rxBuf[2] == 0x02) g_touchEnabled = (rxBuf[3] == 1);
#endif
            portYIELD();
            continue;
        }

        if (rxBuf[1] != 0xBB || n < 8) { portYIELD(); continue; }
        uint8_t  fId     = rxBuf[2];
        uint8_t  cId     = rxBuf[3];
        uint8_t  nChunks = rxBuf[4];
        uint8_t  flags   = rxBuf[5];
        uint16_t fSize   = ((uint16_t)rxBuf[6] << 8) | rxBuf[7];
        int      dataLen = n - 8;
        uint8_t expectedChunks = (uint8_t)((fSize + CHUNK_DATA_SIZE - 1) / CHUNK_DATA_SIZE);
        uint16_t chunkOffset = (uint16_t)cId * CHUNK_DATA_SIZE;
        int expectedLen = (int)fSize - chunkOffset;
        if (expectedLen > CHUNK_DATA_SIZE) expectedLen = CHUNK_DATA_SIZE;
        if (flags != 0 || dataLen <= 0 || fSize < 4 ||
            fSize > MAX_JPEG_FRAME || nChunks == 0 ||
            nChunks != expectedChunks || cId >= nChunks ||
            dataLen != expectedLen) {
            g_networkDrops = g_networkDrops + 1;
            portYIELD();
            continue;
        }

        if (droppedFrameActive && fId != droppedFrameId) {
            droppedFrameActive = false;
        }
        if (droppedFrameActive) {
            portYIELD();
            continue;
        }

        int frameIdx = -1;
        for (uint8_t i = 0; i < g_numJpegSlots; ++i) {
            if (jpegFrames[i].frameId == fId &&
                jpegFrames[i].inUse) {
                frameIdx = i;
                break;
            }
        }

        if (frameIdx < 0) {
            uint8_t freeIdx;
            if (xQueueReceive(jpegFreeQueue, &freeIdx, 0) != pdTRUE) {
                droppedFrameActive = true;
                droppedFrameId = fId;
                g_networkDrops = g_networkDrops + 1;
                g_abortedFrames = g_abortedFrames + 1;
                portYIELD();
                continue;
            }
            JpegFrameSlot& frame = jpegFrames[freeIdx];
            resetFrameSlot(frame);
            frame.frameId = fId;
            frame.totalChunks = nChunks;
            frame.frameSize = fSize;
            frame.firstChunkMs = millis();
            frame.receiving = true;
            frame.inUse = true;
            frameIdx = freeIdx;
        }

        JpegFrameSlot& frame = jpegFrames[frameIdx];
        if (frame.queued) {
            portYIELD();
            continue;
        }
        if (!frame.receiving || frame.totalChunks != nChunks ||
            frame.frameSize != fSize) {
            portYIELD();
            continue;
        }

        if (!frame.chunkReceived[cId]) {
            memcpy(frame.data + chunkOffset, &rxBuf[8], (size_t)dataLen);
            frame.chunkReceived[cId] = true;
            frame.chunksReceived++;
        }

        if (frame.chunksReceived == frame.totalChunks) {
            if (frame.data[0] != 0xFF || frame.data[1] != 0xD8 ||
                frame.data[frame.frameSize - 2] != 0xFF ||
                frame.data[frame.frameSize - 1] != 0xD9) {
                g_networkDrops = g_networkDrops + 1;
                g_abortedFrames = g_abortedFrames + 1;
                returnJpegSlot((uint8_t)frameIdx);
            } else {
                frame.receiving = false;
                frame.queued = true;
            }
        }

        if (debugEnabled && g_remoteAddrValid && (millis() - lastStatMs) > DEBUGINTERVAL) {
            uint32_t el = millis() - lastStatMs;

            static uint32_t lastPresented = 0;
            uint32_t nowPresented = g_presentedFrames;
            uint32_t frames = nowPresented - lastPresented;
            lastPresented = nowPresented;
            float fps = frames / (el / 1000.0f);

            static uint32_t lastNetworkDrops = 0;
            uint32_t networkDrops = g_networkDrops - lastNetworkDrops;
            lastNetworkDrops = g_networkDrops;

            static uint32_t lastAborted = 0;
            uint32_t nowAborted = g_abortedFrames;
            uint32_t aborted = nowAborted - lastAborted;
            lastAborted = nowAborted;

            uint32_t freeSRAM  = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
            uint32_t totalSRAM = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
            uint32_t freePSR   = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
            uint32_t totalPSR  = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
            float    tempC     = readChipTempC();
            uint32_t decUs     = g_avgDecodeUs;

            static uint32_t lastIdleUs[2] = { 0, 0 };
            uint32_t idleNow0   = g_cpuIdleUs[0];
            uint32_t idleNow1   = g_cpuIdleUs[1];
            uint32_t idleDelta0 = idleNow0 - lastIdleUs[0];
            uint32_t idleDelta1 = idleNow1 - lastIdleUs[1];
            lastIdleUs[0] = idleNow0;
            lastIdleUs[1] = idleNow1;
            uint32_t totalUs = el * 1000u;
            uint32_t cpu0 = (totalUs == 0 || idleDelta0 >= totalUs)
                            ? 0 : (100u - idleDelta0 * 100u / totalUs);
            uint32_t cpu1 = (totalUs == 0 || idleDelta1 >= totalUs)
                            ? 0 : (100u - idleDelta1 * 100u / totalUs);

            snprintf(debugBuf, sizeof(debugBuf),
                "%c%cFPS:%.1f|TEMP:%.1f|JIT:%.1f|DEC:%lu|DROP:%lu|ABRT:%lu"
                "|SRAM:%lu/%lu|PSRAM:%lu/%lu|CPU0:%lu|CPU1:%lu",
                0xAB, 0xCD,
                fps, tempC, stat_jitter,
                decUs, networkDrops, aborted,
                freeSRAM / 1024, totalSRAM / 1024,
                freePSR  / 1024, totalPSR  / 1024,
                cpu0, cpu1);

            sendto(g_sock, debugBuf, strlen(debugBuf), 0,
                   (struct sockaddr*)&g_remoteAddr, sizeof(g_remoteAddr));

            lastStatMs = millis();
        }

#if TOUCH_FEATURE_ENABLED
        flushTouchEvents();
#endif
        portYIELD();
    }
}

void wifiWatchdogTask(void*) {
    uint8_t tick = 0;

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(250));

        bool connected = g_wifiOk;

        if (connected) {
            if (!g_streaming) {
                int32_t rssi = wifiGetRSSI();
                char    ip[16];
                wifiGetIPString(ip, sizeof(ip));
                char    buf[40];
                snprintf(buf, sizeof(buf), "%s (%ddBm)", ip, (int)rssi);
                statusLine(3, "WiFi:", buf, TFT_GREEN);
            }
            continue;
        }

        if (!g_streaming) {
            char animBuf[24];
            snprintf(animBuf, sizeof(animBuf), "Connecting%.*s", tick % 4, "....");
            statusLine(3, "WiFi:", animBuf, TFT_YELLOW);
            tick++;

            wifiReconnect();

            uint32_t ws = millis();
            while (!g_wifiOk && (millis() - ws) < 10000) {
                vTaskDelay(pdMS_TO_TICKS(250));
                snprintf(animBuf, sizeof(animBuf), "Connecting%.*s", tick % 4, "....");
                statusLine(3, "WiFi:", animBuf, TFT_YELLOW);
                tick++;
            }

            if (g_wifiOk) {
                esp_wifi_set_ps(WIFI_PS_NONE);
                int32_t rssi = wifiGetRSSI();
                char    ip[16];
                wifiGetIPString(ip, sizeof(ip));
                char    buf[40];
                snprintf(buf, sizeof(buf), "%s (%ddBm)", ip, (int)rssi);
                statusLine(3, "WiFi:", buf, TFT_GREEN);
            }
            continue;
        }

        // Streaming was active, WiFi dropped — reconnect in background
        static uint32_t lastReconnectMs = 0;
        if (millis() - lastReconnectMs >= 3000) {
            lastReconnectMs = millis();
            wifiReconnect();
        }
    }
}
