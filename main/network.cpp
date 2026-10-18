#include "network.h"
#include "display.h"
#include "wifi.h"
#include "nexus.h"
#include "esp_heap_caps.h"
#include "driver/temp_sensor.h"
#include "esp_wifi.h"

/*
 * REBASE NOTE: only two things changed here vs. the Arduino version —
 *  1. WiFi.status()/.RSSI()/.localIP().toString()/.begin()/.disconnect()
 *     became wifiReconnect() + wifiGetRSSI() + wifiGetIPString() (wifi.h),
 *     and g_wifiOk (set asynchronously by wifi.cpp's event handler) instead
 *     of polling WiFi.status() synchronously.
 *  2. Arduino's temperatureRead() became the native ESP-IDF temperature
 *     sensor driver (driver/temp_sensor.h), which needs a one-time install.
 * Everything else — the UDP assembly state machine, chunk reassembly,
 * stats packet, touch event draining — is untouched.
 */

static bool s_tempSensorReady = false;

static float readChipTempC() {
    if (!s_tempSensorReady) {
        temp_sensor_config_t cfg = TSENS_CONFIG_DEFAULT();
        if (temp_sensor_set_config(cfg) != ESP_OK) return 0.0f;
        if (temp_sensor_start() != ESP_OK) return 0.0f;
        s_tempSensorReady = true;
    }
    float tC = 0.0f;
    temp_sensor_read_celsius(&tC);
    return tC;
}

static IRAM_ATTR void resetTile(uint8_t t) {
    memset(tiles[t].chunkGot, 0, sizeof(tiles[t].chunkGot));
    tiles[t].frameId      = 0xFF;
    tiles[t].totalChunks  = 0;
    tiles[t].frameSize    = 0;
    tiles[t].chunksGot    = 0;
    tiles[t].firstChunkMs = 0;
}

static IRAM_ATTR int assembleTileInto(uint8_t t, uint8_t* dst) {
    TileState& ts = tiles[t];
    if (ts.totalChunks == 0) return 0;

    int offset = 0;
    for (uint8_t c = 0; c < ts.totalChunks; c++) {
        if (!ts.chunkGot[c]) return 0;
        memcpy(dst + offset, ts.chunkBuf[c], ts.chunkLen[c]);
        offset += ts.chunkLen[c];
    }

    if (offset < 4 ||
        dst[0] != 0xFF || dst[1] != 0xD8 ||
        dst[offset-2] != 0xFF || dst[offset-1] != 0xD9) {
        ts.stat_corrupt++;
        return 0;
    }
    return offset;
}

// Drains touchTask's queue onto the wire. Called from networkTask's own loop so
// the UDP socket is only ever used by one task.
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

void networkTask(void*) {
    g_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (g_sock < 0) { vTaskDelete(NULL); return; }

    int rcvbuf = 65536;
    setsockopt(g_sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

    struct sockaddr_in local = {};
    local.sin_family      = AF_INET;
    local.sin_port        = htons(UDP_PORT);
    local.sin_addr.s_addr = INADDR_ANY;
    if (bind(g_sock, (struct sockaddr*)&local, sizeof(local)) < 0) {
        close(g_sock);
        vTaskDelete(NULL);
        return;
    }
    fcntl(g_sock, F_SETFL, O_NONBLOCK);

    static uint8_t rxBuf[CHUNK_DATA_SIZE + 16];
    struct sockaddr_in sender;
    socklen_t slen = sizeof(sender);
    uint32_t lastPktMs = millis(), lastBeaconMs = 0, lastStatMs = 0;
    uint8_t back = 0;

    while (true) {
        int n = recvfrom(g_sock, rxBuf, sizeof(rxBuf), 0,
                         (struct sockaddr*)&sender, &slen);

        if (n < 0) {
            flushTouchEvents();

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
            if (n >= 4 && rxBuf[2] == 0x02) g_touchEnabled = (rxBuf[3] == 1);
            portYIELD();
            continue;
        }

        if (rxBuf[1] != 0xBB || n < 8) { portYIELD(); continue; }
        uint8_t  fId     = rxBuf[2];
        uint8_t  tId     = rxBuf[3];
        uint8_t  cId     = rxBuf[4];
        uint8_t  nChunks = rxBuf[5];
        uint16_t fSize   = ((uint16_t)rxBuf[6] << 8) | rxBuf[7];
        int      dataLen = n - 8;
        if (tId >= NUM_TILES || dataLen <= 0) { portYIELD(); continue; }

        TileState& ts = tiles[tId];

        if (ts.firstChunkMs > 0 && (millis() - ts.firstChunkMs) > TILE_TIMEOUT_MS) {
            ts.stat_timeout++;
            resetTile(tId);
        }

        if (fId != ts.frameId) {
            resetTile(tId);
            ts.frameId      = fId;
            ts.totalChunks  = (nChunks > MAX_TILE_CHUNKS) ? MAX_TILE_CHUNKS : nChunks;
            ts.frameSize    = fSize;
            ts.firstChunkMs = millis();
        }

        if (cId < MAX_TILE_CHUNKS && !ts.chunkGot[cId]) {
            memcpy(ts.chunkBuf[cId], &rxBuf[8], dataLen);
            ts.chunkLen[cId] = (uint16_t)dataLen;
            ts.chunkGot[cId] = true;
            ts.chunksGot++;
        }

        if (ts.chunksGot >= ts.totalChunks) {
            xSemaphoreTake(slotFree[back], portMAX_DELAY);
            int len = assembleTileInto(tId, slot[back].assembly);

            if (len > 0) {
                DecodeMsg msg = { fId, tId, back, (uint16_t)len };
                xQueueSend(decodeQueue, &msg, portMAX_DELAY);
                back = (back + 1) % g_numJpegSlots;
            } else {
                xSemaphoreGive(slotFree[back]);
            }
            resetTile(tId);
        }

        if (debugEnabled && g_remoteAddrValid && (millis() - lastStatMs) > DEBUGINTERVAL) {
            uint32_t el = millis() - lastStatMs;

            static uint32_t lastPresented = 0;
            uint32_t nowPresented = g_presentedFrames;
            uint32_t frames = nowPresented - lastPresented;
            lastPresented = nowPresented;
            float fps = frames / (el / 1000.0f);

            uint32_t totalDrop = 0;
            for (int i = 0; i < NUM_TILES; i++)
                totalDrop += tiles[i].stat_corrupt + tiles[i].stat_timeout;

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
                decUs, totalDrop, aborted,
                freeSRAM / 1024, totalSRAM / 1024,
                freePSR  / 1024, totalPSR  / 1024,
                cpu0, cpu1);

            sendto(g_sock, debugBuf, strlen(debugBuf), 0,
                   (struct sockaddr*)&g_remoteAddr, sizeof(g_remoteAddr));

            for (int i = 0; i < NUM_TILES; i++)
                tiles[i].stat_decoded = tiles[i].stat_corrupt = tiles[i].stat_timeout = 0;
            lastStatMs = millis();
        }

        flushTouchEvents();
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
