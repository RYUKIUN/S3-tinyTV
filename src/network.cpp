#include "network.h"
#include "display.h"
#include <WiFi.h>

// Extern declarations for global variables
extern volatile uint32_t g_wifiDisconnectedMs;

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
        // ── Offline-mode shutdown ─────────────────────────────────────────────
        // enterOfflineMode() sets g_offlineMode then closes g_sock.
        // We may still be in recvfrom / select; check here on each wakeup.
        if (g_offlineMode) {
            if (g_sock >= 0) { close(g_sock); g_sock = -1; }
            Serial.println("[NET] Offline mode — networkTask exiting");
            vTaskDelete(NULL);
            return;
        }

        int n = recvfrom(g_sock, rxBuf, sizeof(rxBuf), 0,
                         (struct sockaddr*)&sender, &slen);

        if (n < 0) {
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
            ts.totalChunks  = nChunks;
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

        if (debugEnabled && g_remoteAddrValid && (millis() - lastStatMs) > 400) {
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
            float    tempC     = temperatureRead();
            uint32_t decUs     = g_avgDecodeUs;

            // ── Per-core CPU% ─────────────────────────────────────────────────
            // FreeRTOS runs at configTICK_RATE_HZ (1000 Hz on ESP32-Arduino),
            // so in `el` ms we expect `el` ticks per core.
            // Each idle-hook call ≈ one idle-task iteration; we use the delta
            // over this 400 ms window as a proxy for idle time.
            static uint32_t lastIdleTicks[2] = { 0, 0 };
            uint32_t idleNow0 = g_cpuIdleTicks[0];
            uint32_t idleNow1 = g_cpuIdleTicks[1];
            uint32_t idleDelta0 = idleNow0 - lastIdleTicks[0];
            uint32_t idleDelta1 = idleNow1 - lastIdleTicks[1];
            lastIdleTicks[0] = idleNow0;
            lastIdleTicks[1] = idleNow1;
            uint32_t ticks = (el > 0) ? el : 1;
            uint32_t cpu0 = (idleDelta0 >= ticks) ? 0 : (100 - (idleDelta0 * 100 / ticks));
            uint32_t cpu1 = (idleDelta1 >= ticks) ? 0 : (100 - (idleDelta1 * 100 / ticks));

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

        portYIELD();
    }
}

void wifiWatchdogTask(void*) {
    uint8_t tick = 0;

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(250));

        // ── Offline-mode shutdown ─────────────────────────────────────────────
        if (g_offlineMode) {
            Serial.println("[WATCHDOG] Offline mode — wifiWatchdogTask exiting");
            vTaskDelete(NULL);
            return;
        }

        bool connected = (WiFi.status() == WL_CONNECTED);
        g_wifiOk = connected;

        if (connected) {            if (g_wifiDisconnectedMs == 0) {
                g_wifiDisconnectedMs = millis();
            }            g_wifiDisconnectedMs = 0;  // reset disconnect timer
            if (!g_streaming) {
                int32_t rssi = WiFi.RSSI();
                String  ip   = WiFi.localIP().toString();
                char    buf[40];
                snprintf(buf, sizeof(buf), "%s (%ddBm)", ip.c_str(), (int)rssi);
                statusLine(3, "WiFi:", buf, TFT_GREEN);
            }
            continue;
        }

        if (!g_streaming) {
            char animBuf[24];
            snprintf(animBuf, sizeof(animBuf), "Connecting%.*s", tick % 4, "....");
            statusLine(3, "WiFi:", animBuf, TFT_YELLOW);
            tick++;

            WiFi.disconnect(true);
            vTaskDelay(pdMS_TO_TICKS(100));
            WiFi.begin(WIFI_SSID, WIFI_PASS);

            uint32_t ws = millis();
            while (WiFi.status() != WL_CONNECTED && (millis() - ws) < 10000) {
                vTaskDelay(pdMS_TO_TICKS(250));
                snprintf(animBuf, sizeof(animBuf), "Connecting%.*s", tick % 4, "....");
                statusLine(3, "WiFi:", animBuf, TFT_YELLOW);
                tick++;
            }

            if (WiFi.status() == WL_CONNECTED) {
                g_wifiOk = true;
                esp_wifi_set_ps(WIFI_PS_NONE);
                int32_t rssi = WiFi.RSSI();
                String  ip   = WiFi.localIP().toString();
                char    buf[40];
                snprintf(buf, sizeof(buf), "%s (%ddBm)", ip.c_str(), (int)rssi);
                statusLine(3, "WiFi:", buf, TFT_GREEN);
            }
            continue;
        }

        // Streaming was active, WiFi dropped — reconnect in background
        static uint32_t lastReconnectMs = 0;
        if (millis() - lastReconnectMs >= 3000) {
            lastReconnectMs = millis();
            WiFi.disconnect(true);
            vTaskDelay(pdMS_TO_TICKS(100));
            WiFi.begin(WIFI_SSID, WIFI_PASS);
        }
    }
}