#pragma once
#include <cstdint>

constexpr int AURASYNC_FEATURE_COUNT = 28;
constexpr int AURASYNC_CLASS_COUNT = 4;

const char* AURASYNC_LABELS[AURASYNC_CLASS_COUNT] = {
  "smooth_background",
  "warning_speech",
  "loud_noise",
  "spike_event",
};

const float AURASYNC_FEATURE_MEAN[AURASYNC_FEATURE_COUNT] = {
  4.86006594f, 3.84562564f, 5.64914131f, 6.74455261f, 0.177813023f, 0.136954457f, 0.161201388f, 0.256220996f, 0.278803498f, 0.225787938f, 0.0779862106f, 0.504591465f, 0.127323851f, 0.210187167f, 0.0465514176f, 0.652363658f, 4.22180128f, 3.90279293f, 4.14813948f, 3.71773338f, 3.44706893f, 3.70726633f, 3.0740068f, 2.97250915f, 2.20705771f, 1.81534564f, 1.49867857f, 1.49359882f,
};

const float AURASYNC_FEATURE_STD[AURASYNC_FEATURE_COUNT] = {
  0.695304513f, 0.59590584f, 0.327030212f, 0.0246567205f, 0.186299875f, 0.0948839039f, 0.264651865f, 0.251429915f, 0.236651748f, 0.235588104f, 0.160130486f, 0.319241285f, 0.123037681f, 0.206232846f, 0.0639875904f, 0.155961946f, 2.12572193f, 1.77433705f, 1.53932071f, 1.5220226f, 1.46826982f, 1.54123282f, 1.47206199f, 1.58138907f, 1.60170662f, 1.54251027f, 1.46992636f, 1.56075907f,
};

// Feature order used during training:
// 00: rms_mean
// 01: rms_std
// 02: rms_max
// 03: peak
// 04: crest
// 05: zcr_mean
// 06: sub80_ratio
// 07: low80_300_ratio
// 08: speech300_1000_ratio
// 09: speech1000_3400_ratio
// 10: high3400_8000_ratio
// 11: speech_total_ratio
// 12: centroid_norm
// 13: rolloff85_norm
// 14: flatness
// 15: entropy
// 16: band_0_125_log
// 17: band_125_250_log
// 18: band_250_500_log
// 19: band_500_750_log
// 20: band_750_1000_log
// 21: band_1000_1500_log
// 22: band_1500_2000_log
// 23: band_2000_3000_log
// 24: band_3000_4000_log
// 25: band_4000_5000_log
// 26: band_5000_6000_log
// 27: band_6000_8000_log