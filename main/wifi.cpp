#include "wifi.h"
#include "nexus.h"

#include "nvs_flash.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_log.h"

static const char* TAG = "wifi";
static esp_netif_t* s_staNetif = nullptr;

void nvsInit() {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
}

static void wifiEventHandler(void* /*arg*/, esp_event_base_t base,
                              int32_t id, void* data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        g_wifiOk = false;
        ESP_LOGW(TAG, "disconnected");
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        auto* event = (ip_event_got_ip_t*)data;
        g_wifiOk          = true;
        g_wifiConnectedMs = millis();
        ESP_LOGI(TAG, "got ip: " IPSTR, IP2STR(&event->ip_info.ip));
    }
}

void wifiInit() {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    s_staNetif = esp_netif_create_default_wifi_sta();
    esp_netif_set_hostname(s_staNetif, OTA_HOSTNAME);

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifiEventHandler, nullptr, nullptr));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifiEventHandler, nullptr, nullptr));

    wifi_config_t wifiConfig = {};
    strncpy((char*)wifiConfig.sta.ssid, WIFI_SSID, sizeof(wifiConfig.sta.ssid) - 1);
    strncpy((char*)wifiConfig.sta.password, WIFI_PASS, sizeof(wifiConfig.sta.password) - 1);
    wifiConfig.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifiConfig));
    ESP_ERROR_CHECK(esp_wifi_start());

    // Equivalent of the old WiFi.setSleep(false)/esp_wifi_set_max_tx_power /
    // esp_wifi_set_protocol / esp_wifi_set_bandwidth calls that used to sit
    // in main.cpp's setup() right after WiFi.begin() — those were already
    // native esp_wifi_* calls, so they just move here unchanged.
    esp_wifi_set_ps(WIFI_PS_NONE);
    esp_wifi_set_max_tx_power(80);   // 0.25 dBm units -> requests 20 dBm;
                                     // actual radiated power may be clamped
                                     // lower by regional regulatory limits
    esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11N);
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW20));
}

void wifiReconnect() {
    esp_wifi_disconnect();
    esp_wifi_connect();
}

void wifiGetIPString(char* buf, size_t len) {
    buf[0] = '\0';
    if (!s_staNetif) return;
    esp_netif_ip_info_t ipInfo;
    if (esp_netif_get_ip_info(s_staNetif, &ipInfo) == ESP_OK) {
        esp_ip4addr_ntoa(&ipInfo.ip, buf, (int)len);
    }
}

int32_t wifiGetRSSI() {
    wifi_ap_record_t ap;
    if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) return ap.rssi;
    return 0;
}
