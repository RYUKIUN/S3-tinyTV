#pragma once
/*
 * dsps_fft2r_platform.h — shim to unlock JPEGDEC's ESP32-S3 SIMD path
 *                          without requiring the full esp-dsp component.
 *
 * JPEGDEC gates its s3_simd_420.S / s3_simd_dequant.S assembly files on:
 *   #if (dsps_fft2r_sc16_aes3_enabled == 1)
 *
 * The real header comes from esp-dsp and probes the platform at compile time.
 * Since Arduino framework for ESP32-S3 already ships with the PIE/AES3 toolchain
 * support (CONFIG_IDF_TARGET_ESP32S3 guarantees the xtensa-esp32s3-elf assembler
 * understands EE.xxx opcodes), we can safely hardcode the flag to 1.
 */
#ifdef CONFIG_IDF_TARGET_ESP32S3
  #define dsps_fft2r_sc16_aes3_enabled 1
#else
  #define dsps_fft2r_sc16_aes3_enabled 0
#endif