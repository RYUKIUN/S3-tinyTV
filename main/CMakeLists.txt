# "main" component — this is where every one of this project's own .cpp
# files is registered. If you add a new .cpp file under main/, add it to
# SRCS below or the build simply won't see it (ESP-IDF does not glob by
# default the way PlatformIO used to).

idf_component_register(
    SRCS
        "app_main.cpp"
        "wifi.cpp"
        "ota.cpp"
        "display.cpp"
        "network.cpp"
        "jpeg_decode.cpp"
        "touch.cpp"
    INCLUDE_DIRS
        "."
    REQUIRES
        nvs_flash
        esp_wifi
        esp_netif
        esp_event
        esp_timer
        esp_psram
        driver
        esp_http_server
        app_update
        esp_hw_support
        heap
        lwip
        esp_system
        # Managed/vendored components pulled in by main/idf_component.yml —
        # the names below MUST match the dependency keys in that file.
        lovyangfx
        esp_new_jpeg
)

# NOTE: if idf.py reports "could not resolve component 'X'" for something
# like esp_task_wdt, esp_phy, vfs, etc. on your installed IDF version, just
# add it to the REQUIRES list above and re-run the build — Kconfig/component
# naming has shifted a little between IDF 5.1/5.3/5.4, and this is the one
# place that absorbs that drift.
