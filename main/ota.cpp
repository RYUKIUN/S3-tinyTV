#include "ota.h"
#include "nexus.h"

#include "esp_http_server.h"
#include "esp_ota_ops.h"
#include "esp_log.h"

static const char* TAG = "ota";

static esp_err_t updateHandler(httpd_req_t* req) {
    const esp_partition_t* target = esp_ota_get_next_update_partition(nullptr);
    if (!target) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "no OTA partition");
        return ESP_FAIL;
    }

    esp_ota_handle_t otaHandle = 0;
    esp_err_t err = esp_ota_begin(target, OTA_SIZE_UNKNOWN, &otaHandle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_ota_begin failed: %s", esp_err_to_name(err));
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "ota_begin failed");
        return ESP_FAIL;
    }

    char     buf[1024];
    int      remaining = req->content_len;
    uint32_t total     = 0;

    ESP_LOGI(TAG, "OTA start -> %s, %d bytes incoming", target->label, remaining);

    while (remaining > 0) {
        int toRead = remaining > (int)sizeof(buf) ? (int)sizeof(buf) : remaining;
        int n = httpd_req_recv(req, buf, toRead);
        if (n <= 0) {
            if (n == HTTPD_SOCK_ERR_TIMEOUT) continue;
            esp_ota_abort(otaHandle);
            httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "recv failed");
            return ESP_FAIL;
        }

        err = esp_ota_write(otaHandle, buf, n);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "esp_ota_write failed: %s", esp_err_to_name(err));
            esp_ota_abort(otaHandle);
            httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "ota_write failed");
            return ESP_FAIL;
        }

        remaining -= n;
        total     += n;
    }

    err = esp_ota_end(otaHandle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_ota_end failed: %s (image validation failed?)", esp_err_to_name(err));
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "ota_end failed");
        return ESP_FAIL;
    }

    err = esp_ota_set_boot_partition(target);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_ota_set_boot_partition failed: %s", esp_err_to_name(err));
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "set_boot_partition failed");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "OTA OK — %u bytes written to %s, rebooting", (unsigned)total, target->label);
    httpd_resp_sendstr(req, "OK - rebooting\n");

    vTaskDelay(pdMS_TO_TICKS(300));   // let the response actually flush
    esp_restart();
    return ESP_OK;   // unreachable
}

void otaStart() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port    = OTA_HTTP_PORT;
    // Firmware images comfortably exceed the default recv buffer; raise it
    // so httpd_req_recv() isn't starved on a slow link.
    config.recv_wait_timeout = 10;
    config.send_wait_timeout = 10;

    httpd_handle_t server = nullptr;
    if (httpd_start(&server, &config) != ESP_OK) {
        ESP_LOGE(TAG, "httpd_start failed — OTA endpoint not available");
        return;
    }

    httpd_uri_t updateUri = {
        .uri      = "/update",
        .method   = HTTP_POST,
        .handler  = updateHandler,
        .user_ctx = nullptr,
    };
    httpd_register_uri_handler(server, &updateUri);

    ESP_LOGI(TAG, "OTA ready: POST firmware.bin to http://<device-ip>:%d/update", OTA_HTTP_PORT);
}
