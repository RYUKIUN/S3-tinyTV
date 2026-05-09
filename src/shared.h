#pragma once

#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <JPEGDEC.h>
#include <LovyanGFX.hpp>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include <lwip/sockets.h>
#include <lwip/netdb.h>
#include <fcntl.h>
#include <math.h>

#define PKT_TIMEOUT_MS   3000
#define OVERLAY_FLASH_MS 1000

#define SCREEN_W         320
#define SCREEN_H         240
#define NUM_TILES        4
#define TILE_W           160
#define TILE_H           120
#define TILE_PIXELS      (TILE_W * TILE_H)
#define CHUNK_DATA_SIZE  1400
#define MAX_TILE_CHUNKS  24
#define MAX_TILE_JPEG    (MAX_TILE_CHUNKS * CHUNK_DATA_SIZE)
#define TILE_TIMEOUT_MS  200

static const int16_t TILE_X[NUM_TILES] = {  0, 160,   0, 160 };
static const int16_t TILE_Y[NUM_TILES] = {  0,   0, 120, 120 };

#define NUM_SLOTS 4

struct PipeSlot {
    uint8_t* assembly;
};

struct DecodeMsg {
    uint8_t  frameId;
    uint8_t  tId;
    uint8_t  slotIdx;
    uint16_t len;
};

struct DisplayMsg {
    uint8_t frameId;
    uint8_t bufSet;
};

struct TileState {
    uint8_t* chunkBuf[MAX_TILE_CHUNKS];
    uint16_t chunkLen[MAX_TILE_CHUNKS];
    bool     chunkGot[MAX_TILE_CHUNKS];
    uint8_t  frameId      = 0xFF;
    uint8_t  totalChunks  = 0;
    uint16_t frameSize    = 0;
    uint8_t  chunksGot    = 0;
    uint32_t firstChunkMs = 0;
    uint32_t stat_decoded = 0;
    uint32_t stat_corrupt = 0;
    uint32_t stat_timeout = 0;
};

extern QueueHandle_t     decodeQueue;
extern QueueHandle_t     displayQueue;
extern SemaphoreHandle_t slotFree[NUM_SLOTS];

extern uint16_t* decodeTemp;
extern uint16_t* frameFb[2];
extern uint8_t* tileChunkStorage[NUM_TILES];
extern PipeSlot slot[NUM_SLOTS];
extern TileState tiles[NUM_TILES];

extern volatile uint32_t g_avgDecodeUs;
extern volatile uint32_t g_presentedFrames;
extern volatile uint32_t g_abortedFrames;

extern volatile bool g_streaming;
extern volatile bool g_wifiOk;
extern volatile uint32_t g_lastPktMs;

extern const char* WIFI_SSID;
extern const char* WIFI_PASS;
extern int UDP_PORT;

extern bool debugEnabled;
extern char debugBuf[256];
extern int g_sock;
extern struct sockaddr_in g_remoteAddr;
extern bool g_remoteAddrValid;
extern float stat_jitter;

extern uint8_t writeSet;
