#include <WiFi.h>
#include <ESPmDNS.h>
#include <WebServer.h>
#include <Preferences.h>
#include "esp_heap_caps.h"


#include "driver/i2s_std.h"
i2s_chan_handle_t rx_handle = NULL;


#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"


#include "model_data.h"


// ==============================================================
// AuraSync ESP32-S3 N8R8 + INMP441 FINAL DEPLOYABLE FIRMWARE
// Pins used by your wiring:
// INMP441 VDD  -> 3.3V
// INMP441 GND  -> GND
// INMP441 SCK  -> GPIO 14
// INMP441 WS   -> GPIO 15
// INMP441 SD   -> GPIO 16
// INMP441 L/R  -> GND, therefore LEFT channel is selected below.
//
// Main improvements over the previous test sketch:
// 1) Safer INMP441 sample conversion: lower clipping and better speech level.
// 2) DC removal + high-pass monitor audio, reducing "kir-kir/click" noise.
// 3) Sequenced live-audio blocks, so browser no longer repeats/skips blocks.
// 4) 10 sec WAV recording and download.
// 5) Noise-floor/SNR based spike detection to stop constant false spikes.
// 6) Runtime dB calibration endpoint stored in ESP32 NVS Preferences.
// 7) CNN window is now chronologically ordered before feature extraction.
// 8) CNN output is displayed but not trusted for critical state by default.
// ==============================================================


// ======================= WIFI =======================
const char* ssid = "PlanAP-HP02_X";
const char* password = "0987654321";


WebServer server(80);
Preferences prefs;


// ======================= I2S MIC PINS =======================
#define I2S_WS   15
#define I2S_SD   16
#define I2S_SCK  14


// ======================= AUDIO SETTINGS =======================
#define SAMPLE_RATE       16000
#define DSP_BLOCK_MS      100
#define DSP_BLOCK_SAMPLES 1600
#define CNN_WINDOW_SEC    1
#define CNN_SAMPLES       16000
#define I2S_READ_CHUNK    512


// INMP441 usually arrives as 24-bit data inside a 32-bit slot.
// For 16-bit WAV/monitor audio, shifting by 16 is safer than 14.
// If your WAV is too quiet after testing, increase MIC_SOFTWARE_GAIN first.
#define MIC_SHIFT_16BIT        16
#define MIC_SOFTWARE_GAIN      2.0f
#define MONITOR_GAIN           1.5f
#define HIGHPASS_ALPHA         0.995f


// Use cleaned/high-pass monitor audio for listening and recording.
// Set to 0 if you truly need unfiltered raw converted PCM.
#define USE_CLEAN_AUDIO_FOR_RECORD 1
#define USE_CLEAN_AUDIO_FOR_LIVE   1


// ======================= RECORD + LIVE BUFFER SETTINGS =======================
#define RECORD_SECONDS      10
#define RECORD_SAMPLES      (SAMPLE_RATE * RECORD_SECONDS)
#define RECORD_BYTES        (RECORD_SAMPLES * sizeof(int16_t))


#define LIVE_RING_BLOCKS_TARGET 60   // 6 seconds of 100 ms blocks. PSRAM preferred.
#define LIVE_FETCH_MAX_BLOCKS   6


// ======================= CALIBRATION =======================
// A digital MEMS mic cannot give true legal/certified dBA without calibration.
// Use dashboard calibration with a real sound meter.
float CAL_OFFSET_DB = 94.0f;


// ======================= STATE THRESHOLDS =======================
// These are intentionally more conservative than the previous version.
const float SMOOTH_DB_MAX       = 58.0f;
const float WARNING_DB_MIN      = 60.0f;
const float WARNING_DB_MAX      = 72.0f;
const float LOUD_ENTER_DB       = 76.0f;
const float LOUD_EXIT_DB        = 72.0f;


// Spike is now SNR + peak + crest-factor based, not only absolute dB.
const float SPIKE_PEAK_ABOVE_NOISE_DB = 22.0f;
const float SPIKE_MIN_PEAK_DB         = 78.0f;
const float SPIKE_MIN_CREST_DB        = 10.0f;


// CNN can be noisy unless the model exactly matches this preprocessing.
// Keep 0 for deployment safety. Set to 1 only after retraining/validation.
#define USE_CNN_IN_DECISION 0
const float CNN_SPEECH_TH       = 0.70f;
const float CNN_LOUD_TH         = 0.75f;
const float CNN_SPIKE_TH        = 0.80f;
const float CNN_QUIET_TH        = 0.45f;


const int PREWARNING_CONFIRM_MS = 1500;
const int WARNING_CONFIRM_MS    = 3500;
const int LOUD_CONFIRM_MS       = 2500;
const int RECOVERY_CONFIRM_MS   = 6000;
const int SPIKE_CONFIRM_MS      = 250;
const int SPIKE_VISIBLE_MS      = 600;
const int MIC_ERROR_CONFIRM_MS  = 2000;


// ======================= CNN CLASS INDEX =======================
#define CLASS_SMOOTH          0
#define CLASS_WARNING_SPEECH  1
#define CLASS_LOUD            2
#define CLASS_SPIKE           3


// ======================= MODEL FEATURE SETTINGS =======================
#define MODEL_TIME_AXIS_FIRST 1
const float FEATURE_LOG_MIN = -12.0f;
const float FEATURE_LOG_MAX = 4.0f;


// ======================= AUDIO BUFFERS =======================
int32_t i2sRaw[I2S_READ_CHUNK];
int16_t rawBlock[DSP_BLOCK_SAMPLES];
int16_t dspBlock[DSP_BLOCK_SAMPLES];       // cleaned block used for metrics/model
int16_t monitorBlock[DSP_BLOCK_SAMPLES];   // listenable PCM for dashboard
int16_t cnnWindow[CNN_SAMPLES];
int16_t cnnOrdered[CNN_SAMPLES];


int cnnWriteIndex = 0;
int dspSamplesFilled = 0;


// HPF state
float hpLastX = 0.0f;
float hpLastY = 0.0f;


// Record buffer
int16_t* recordBuffer = NULL;
volatile bool isRecording = false;
volatile bool recordReady = false;
uint32_t recordWriteSamples = 0;
unsigned long recordStartedMs = 0;
unsigned long recordFinishedMs = 0;


// Live sequenced ring buffer
int16_t* liveRing = NULL;
uint16_t liveRingBlocks = 0;
volatile uint32_t liveSeqNext = 0;  // next sequence number to be written
volatile bool liveReady = false;


// ======================= TFLITE GLOBALS =======================
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* inputTensor = nullptr;
TfLiteTensor* outputTensor = nullptr;
bool tinyMlOk = false;


constexpr int kTensorArenaSize = 120 * 1024;
alignas(16) uint8_t tensorArena[kTensorArenaSize];


// ======================= METRICS =======================
float g_spl = 0.0f;
float g_splFast = 0.0f;
float g_peakDb = 0.0f;
float g_rmsDbFs = -120.0f;
float g_peakDbFs = -120.0f;
float g_noiseDbFs = -80.0f;
float g_snrDb = 0.0f;
float g_crestDb = 0.0f;
float g_speechRatio = 0.0f;
float g_clipPct = 0.0f;
float g_zeroPct = 0.0f;
float g_dcOffset = 0.0f;
float g_micQuality = 0.0f;


float g_cnnSmooth = 0.0f;
float g_cnnSpeech = 0.0f;
float g_cnnLoud = 0.0f;
float g_cnnSpike = 0.0f;


unsigned long lastDspMs = 0;
unsigned long lastCnnMs = 0;
unsigned long spikeUntilMs = 0;


int prewarningMs = 0;
int warningMs = 0;
int loudMs = 0;
int quietMs = 0;
int spikeCandidateMs = 0;
int micErrorMs = 0;


int eventCount = 0;
int spikeCount = 0;
int quietStreakSec = 0;
uint32_t i2sErrorCount = 0;


// ======================= STATES =======================
enum AuraState {
  STATE_MIC_ERROR,
  STATE_SMOOTH,
  STATE_PREWARNING,
  STATE_WARNING,
  STATE_LOUD,
  STATE_RECOVERY,
  STATE_SPIKE_OVERLAY
};


AuraState stableState = STATE_SMOOTH;
AuraState displayState = STATE_SMOOTH;


const char* stateName(AuraState s) {
  switch (s) {
    case STATE_MIC_ERROR: return "MIC / WIRING ERROR";
    case STATE_SMOOTH: return "SMOOTH";
    case STATE_PREWARNING: return "PRE-WARNING";
    case STATE_WARNING: return "WARNING";
    case STATE_LOUD: return "LOUD";
    case STATE_RECOVERY: return "RECOVERY";
    case STATE_SPIKE_OVERLAY: return "SPIKE";
    default: return "UNKNOWN";
  }
}


// ======================= SAFE SAMPLE HELPERS =======================
int16_t clamp16(int32_t v) {
  if (v > 32767) return 32767;
  if (v < -32768) return -32768;
  return (int16_t)v;
}


int16_t convertSample(int32_t raw) {
  // Safer 24-bit-in-32-bit to 16-bit conversion.
  // Old MIC_SHIFT 14 over-amplified and often clipped noisy samples.
  int32_t sample = raw >> MIC_SHIFT_16BIT;
  sample = (int32_t)(sample * MIC_SOFTWARE_GAIN);
  return clamp16(sample);
}


int16_t highPassAndMonitor(int16_t raw16) {
  float x = (float)raw16;
  float y = x - hpLastX + HIGHPASS_ALPHA * hpLastY;
  hpLastX = x;
  hpLastY = y;


  int32_t out = (int32_t)(y * MONITOR_GAIN);
  return clamp16(out);
}


// ======================= I2S SETUP =======================
bool setupI2S() {
  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  esp_err_t err = i2s_new_channel(&chan_cfg, NULL, &rx_handle);


  if (err != ESP_OK) {
    Serial.print("I2S channel creation failed: ");
    Serial.println((int)err);
    return false;
  }


  i2s_std_config_t std_cfg = {
      .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
      .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
      .gpio_cfg = {
          .mclk = I2S_GPIO_UNUSED,
          .bclk = (gpio_num_t)I2S_SCK,
          .ws   = (gpio_num_t)I2S_WS,
          .dout = I2S_GPIO_UNUSED,
          .din  = (gpio_num_t)I2S_SD,
          .invert_flags = {
              .mclk_inv = false,
              .bclk_inv = false,
              .ws_inv   = false,
          },
      },
  };


  // L/R is tied to GND in your wiring, so use LEFT.
  std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;


  err = i2s_channel_init_std_mode(rx_handle, &std_cfg);
  if (err != ESP_OK) {
    Serial.print("I2S channel init failed: ");
    Serial.println((int)err);
    return false;
  }


  err = i2s_channel_enable(rx_handle);
  if (err != ESP_OK) {
    Serial.print("I2S channel enable failed: ");
    Serial.println((int)err);
    return false;
  }


  Serial.println("I2S microphone initialized.");
  return true;
}


// ======================= MEMORY ALLOCATION =======================
bool allocateLiveRing() {
  if (liveRing != NULL) return true;


  liveRingBlocks = LIVE_RING_BLOCKS_TARGET;
  size_t bytes = (size_t)liveRingBlocks * DSP_BLOCK_SAMPLES * sizeof(int16_t);


  liveRing = (int16_t*)heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);


  if (liveRing == NULL) {
    // Small fallback for boards with weak PSRAM configuration.
    liveRingBlocks = 12;
    bytes = (size_t)liveRingBlocks * DSP_BLOCK_SAMPLES * sizeof(int16_t);
    liveRing = (int16_t*)malloc(bytes);
  }


  if (liveRing == NULL) {
    liveRingBlocks = 0;
    Serial.println("Live ring allocation failed. Live audio disabled.");
    return false;
  }


  memset(liveRing, 0, bytes);
  Serial.print("Live ring allocated bytes: ");
  Serial.println(bytes);
  return true;
}


bool ensureRecordBuffer() {
  if (recordBuffer != NULL) return true;


  recordBuffer = (int16_t*)heap_caps_malloc(RECORD_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);


  if (recordBuffer == NULL) {
    recordBuffer = (int16_t*)malloc(RECORD_BYTES);
  }


  if (recordBuffer == NULL) {
    Serial.println("Record buffer allocation failed. Need about 320 KB free RAM/PSRAM.");
    return false;
  }


  memset(recordBuffer, 0, RECORD_BYTES);
  Serial.print("Record buffer allocated bytes: ");
  Serial.println(RECORD_BYTES);
  return true;
}


int16_t* liveBlockPtr(uint32_t seq) {
  if (liveRing == NULL || liveRingBlocks == 0) return NULL;
  return liveRing + ((seq % liveRingBlocks) * DSP_BLOCK_SAMPLES);
}


void pushLiveBlock(const int16_t* block) {
  if (liveRing == NULL || liveRingBlocks == 0) return;


  uint32_t seq = liveSeqNext;
  int16_t* dst = liveBlockPtr(seq);
  if (dst != NULL) {
    memcpy(dst, block, DSP_BLOCK_SAMPLES * sizeof(int16_t));
    liveSeqNext = seq + 1;
    liveReady = true;
  }
}


// ======================= TFLITE SETUP =======================
void setupTinyML() {
  model = tflite::GetModel(g_aurasync_state_model);


  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch. CNN disabled.");
    tinyMlOk = false;
    return;
  }


  static tflite::AllOpsResolver resolver;
  static tflite::MicroErrorReporter micro_error_reporter;


  static tflite::MicroInterpreter staticInterpreter(
    model,
    resolver,
    tensorArena,
    kTensorArenaSize,
    &micro_error_reporter,
    nullptr,
    nullptr
  );


  interpreter = &staticInterpreter;


  TfLiteStatus allocateStatus = interpreter->AllocateTensors();
  if (allocateStatus != kTfLiteOk) {
    Serial.println("AllocateTensors failed. CNN disabled. Increase tensor arena or check model ops.");
    tinyMlOk = false;
    return;
  }


  inputTensor = interpreter->input(0);
  outputTensor = interpreter->output(0);
  tinyMlOk = true;


  Serial.println("TinyML model loaded.");
  Serial.print("Input tensor bytes: ");
  Serial.println(inputTensor->bytes);
  Serial.print("Output tensor bytes: ");
  Serial.println(outputTensor->bytes);
}


// ======================= READ AUDIO =======================
void readAudio100msBlock() {
  dspSamplesFilled = 0;


  while (dspSamplesFilled < DSP_BLOCK_SAMPLES) {
    size_t bytesRead = 0;


    esp_err_t result = i2s_channel_read(
      rx_handle,
      i2sRaw,
      sizeof(i2sRaw),
      &bytesRead,
      250 / portTICK_PERIOD_MS
    );


    if (result != ESP_OK || bytesRead == 0) {
      i2sErrorCount++;
      delay(1);
      continue;
    }


    int samplesRead = bytesRead / sizeof(int32_t);


    for (int i = 0; i < samplesRead && dspSamplesFilled < DSP_BLOCK_SAMPLES; i++) {
      int16_t raw16 = convertSample(i2sRaw[i]);
      int16_t clean16 = highPassAndMonitor(raw16);


      rawBlock[dspSamplesFilled] = raw16;
      dspBlock[dspSamplesFilled] = clean16;
      monitorBlock[dspSamplesFilled] = clean16;


      cnnWindow[cnnWriteIndex] = clean16;
      cnnWriteIndex++;
      if (cnnWriteIndex >= CNN_SAMPLES) cnnWriteIndex = 0;


      dspSamplesFilled++;
    }
  }
}


// ======================= GOERTZEL ENERGY =======================
float goertzelEnergy(const int16_t* data, int start, int length, float targetFreq) {
  float omega = 2.0f * PI * targetFreq / SAMPLE_RATE;
  float coeff = 2.0f * cos(omega);


  float q0 = 0;
  float q1 = 0;
  float q2 = 0;


  for (int i = 0; i < length; i++) {
    int idx = start + i;
    float sample = data[idx] / 32768.0f;


    float w = 0.54f - 0.46f * cos((2.0f * PI * i) / (length - 1));
    sample *= w;


    q0 = coeff * q1 - q2 + sample;
    q2 = q1;
    q1 = q0;
  }


  float power = q1 * q1 + q2 * q2 - coeff * q1 * q2;
  if (power < 0) power = 0;
  return power;
}


// ======================= DSP FEATURES =======================
void computeDSP() {
  double sumSquares = 0.0;
  double sumRaw = 0.0;
  int32_t peakAbs = 0;
  int32_t clipCount = 0;
  int32_t zeroCount = 0;


  for (int i = 0; i < DSP_BLOCK_SAMPLES; i++) {
    int32_t s = dspBlock[i];
    int32_t r = rawBlock[i];
    int32_t a = abs(s);


    if (a > peakAbs) peakAbs = a;
    if (r > 32000 || r < -32000) clipCount++;
    if (r > -4 && r < 4) zeroCount++;


    float x = s / 32768.0f;
    sumSquares += x * x;
    sumRaw += r;
  }


  float rms = sqrt(sumSquares / DSP_BLOCK_SAMPLES);
  if (rms < 0.000001f) rms = 0.000001f;
  if (peakAbs < 1) peakAbs = 1;


  g_rmsDbFs = 20.0f * log10(rms);
  g_peakDbFs = 20.0f * log10(peakAbs / 32768.0f);
  g_splFast = g_rmsDbFs + CAL_OFFSET_DB;


  // Slow SPL smoothing for stable display.
  static bool splStarted = false;
  if (!splStarted) {
    g_spl = g_splFast;
    splStarted = true;
  } else {
    g_spl = 0.85f * g_spl + 0.15f * g_splFast;
  }


  g_peakDb = g_peakDbFs + CAL_OFFSET_DB;
  g_crestDb = g_peakDbFs - g_rmsDbFs;


  g_clipPct = 100.0f * clipCount / DSP_BLOCK_SAMPLES;
  g_zeroPct = 100.0f * zeroCount / DSP_BLOCK_SAMPLES;
  g_dcOffset = (float)(sumRaw / DSP_BLOCK_SAMPLES);


  // Mic quality: 100 means healthy signal. Low means loose wire, wrong channel, clipping, or no data.
  g_micQuality = 100.0f;
  if (g_zeroPct > 80.0f) g_micQuality -= 60.0f;
  if (g_clipPct > 2.0f) g_micQuality -= 35.0f;
  if (g_rmsDbFs < -75.0f) g_micQuality -= 45.0f;
  if (abs((int)g_dcOffset) > 12000) g_micQuality -= 25.0f;
  if (g_micQuality < 0.0f) g_micQuality = 0.0f;


  // Noise floor learner. It only learns during low-energy/non-spike periods.
  bool likelyQuiet = (g_splFast < WARNING_DB_MIN && g_clipPct < 1.0f && g_crestDb < 18.0f);
  if (likelyQuiet) {
    g_noiseDbFs = 0.995f * g_noiseDbFs + 0.005f * g_rmsDbFs;
  }
  g_snrDb = g_rmsDbFs - g_noiseDbFs;
  if (g_snrDb < 0.0f) g_snrDb = 0.0f;


  // Approximate speech-band ratio.
  float speechEnergy = 0;
  speechEnergy += goertzelEnergy(dspBlock, 0, 512, 400);
  speechEnergy += goertzelEnergy(dspBlock, 0, 512, 800);
  speechEnergy += goertzelEnergy(dspBlock, 0, 512, 1200);
  speechEnergy += goertzelEnergy(dspBlock, 0, 512, 2000);
  speechEnergy += goertzelEnergy(dspBlock, 0, 512, 3200);


  float otherEnergy = 0;
  otherEnergy += goertzelEnergy(dspBlock, 0, 512, 80);
  otherEnergy += goertzelEnergy(dspBlock, 0, 512, 150);
  otherEnergy += goertzelEnergy(dspBlock, 0, 512, 5000);
  otherEnergy += goertzelEnergy(dspBlock, 0, 512, 7000);


  g_speechRatio = speechEnergy / (speechEnergy + otherEnergy + 0.000001f);
  if (g_speechRatio < 0) g_speechRatio = 0;
  if (g_speechRatio > 1) g_speechRatio = 1;
}


// ======================= MEL SCALE HELPERS =======================
float hzToMel(float hz) {
  return 2595.0f * log10(1.0f + hz / 700.0f);
}


float melToHz(float mel) {
  return 700.0f * (pow(10.0f, mel / 2595.0f) - 1.0f);
}


float normalizeFeature(float x) {
  float n = (x - FEATURE_LOG_MIN) / (FEATURE_LOG_MAX - FEATURE_LOG_MIN);
  if (n < 0) n = 0;
  if (n > 1) n = 1;
  return n;
}


void setInputFeature(int index, float featureValue) {
  if (inputTensor == nullptr) return;


  if (inputTensor->type == kTfLiteFloat32) {
    inputTensor->data.f[index] = featureValue;
  }
  else if (inputTensor->type == kTfLiteInt8) {
    int32_t q = round(featureValue / inputTensor->params.scale + inputTensor->params.zero_point);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    inputTensor->data.int8[index] = (int8_t)q;
  }
  else if (inputTensor->type == kTfLiteUInt8) {
    int32_t q = round(featureValue / inputTensor->params.scale + inputTensor->params.zero_point);
    if (q > 255) q = 255;
    if (q < 0) q = 0;
    inputTensor->data.uint8[index] = (uint8_t)q;
  }
}


// ======================= CNN FEATURE EXTRACTION =======================
void extractFeaturesToTensor() {
  if (inputTensor == nullptr) return;


  // Important fix: cnnWindow is circular. Convert it to chronological order.
  for (int i = 0; i < CNN_SAMPLES; i++) {
    int idx = cnnWriteIndex + i;
    if (idx >= CNN_SAMPLES) idx -= CNN_SAMPLES;
    cnnOrdered[i] = cnnWindow[idx];
  }


  int dimsCount = inputTensor->dims->size;
  int totalElements = 1;
  for (int i = 0; i < dimsCount; i++) totalElements *= inputTensor->dims->data[i];


  int timeFrames = 1;
  int freqBins = totalElements;


  if (dimsCount == 4) {
#if MODEL_TIME_AXIS_FIRST
    timeFrames = inputTensor->dims->data[1];
    freqBins = inputTensor->dims->data[2];
#else
    freqBins = inputTensor->dims->data[1];
    timeFrames = inputTensor->dims->data[2];
#endif
  }
  else if (dimsCount == 3) {
    timeFrames = inputTensor->dims->data[1];
    freqBins = inputTensor->dims->data[2];
  }
  else if (dimsCount == 2) {
    freqBins = inputTensor->dims->data[1];
    timeFrames = 1;
  }


  if (timeFrames < 1) timeFrames = 1;
  if (freqBins < 1) freqBins = 1;


  const int frameLen = 512;
  int hop = 0;
  if (timeFrames > 1) hop = (CNN_SAMPLES - frameLen) / (timeFrames - 1);


  float melMin = hzToMel(80);
  float melMax = hzToMel(7600);
  int featureIndex = 0;


  for (int t = 0; t < timeFrames; t++) {
    int start = t * hop;
    if (start < 0) start = 0;
    if (start + frameLen > CNN_SAMPLES) start = CNN_SAMPLES - frameLen;


    for (int b = 0; b < freqBins; b++) {
      float mel = melMin + (melMax - melMin) * (b + 1) / (freqBins + 1);
      float freq = melToHz(mel);
      float energy = goertzelEnergy(cnnOrdered, start, frameLen, freq);
      float logEnergy = log(energy + 0.000001f);
      float normalized = normalizeFeature(logEnergy);


      if (featureIndex < totalElements) setInputFeature(featureIndex, normalized);
      featureIndex++;
    }
  }


  while (featureIndex < totalElements) {
    setInputFeature(featureIndex, 0.0f);
    featureIndex++;
  }
}


float getOutputScoreRaw(int index) {
  if (outputTensor == nullptr) return 0.0f;


  int outputCount = 1;
  for (int i = 0; i < outputTensor->dims->size; i++) outputCount *= outputTensor->dims->data[i];
  if (index < 0 || index >= outputCount) return 0.0f;


  if (outputTensor->type == kTfLiteFloat32) return outputTensor->data.f[index];
  if (outputTensor->type == kTfLiteInt8) {
    int8_t q = outputTensor->data.int8[index];
    return (q - outputTensor->params.zero_point) * outputTensor->params.scale;
  }
  if (outputTensor->type == kTfLiteUInt8) {
    uint8_t q = outputTensor->data.uint8[index];
    return (q - outputTensor->params.zero_point) * outputTensor->params.scale;
  }
  return 0.0f;
}


void normalizeCnnOutputs(float raw0, float raw1, float raw2, float raw3) {
  float vals[4] = { raw0, raw1, raw2, raw3 };
  float sum = raw0 + raw1 + raw2 + raw3;
  bool looksProbability = true;
  for (int i = 0; i < 4; i++) {
    if (vals[i] < -0.01f || vals[i] > 1.01f) looksProbability = false;
  }
  if (sum < 0.70f || sum > 1.30f) looksProbability = false;


  if (looksProbability) {
    if (sum < 0.0001f) sum = 1.0f;
    g_cnnSmooth = vals[0] / sum;
    g_cnnSpeech = vals[1] / sum;
    g_cnnLoud = vals[2] / sum;
    g_cnnSpike = vals[3] / sum;
    return;
  }


  // Fallback: treat model outputs as logits and softmax them.
  float maxVal = vals[0];
  for (int i = 1; i < 4; i++) if (vals[i] > maxVal) maxVal = vals[i];
  float e0 = exp(vals[0] - maxVal);
  float e1 = exp(vals[1] - maxVal);
  float e2 = exp(vals[2] - maxVal);
  float e3 = exp(vals[3] - maxVal);
  float es = e0 + e1 + e2 + e3 + 0.000001f;


  g_cnnSmooth = e0 / es;
  g_cnnSpeech = e1 / es;
  g_cnnLoud = e2 / es;
  g_cnnSpike = e3 / es;
}


void runCNN() {
  if (!tinyMlOk || interpreter == nullptr || inputTensor == nullptr || outputTensor == nullptr) return;


  extractFeaturesToTensor();


  TfLiteStatus invokeStatus = interpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("CNN Invoke failed");
    return;
  }


  float r0 = getOutputScoreRaw(CLASS_SMOOTH);
  float r1 = getOutputScoreRaw(CLASS_WARNING_SPEECH);
  float r2 = getOutputScoreRaw(CLASS_LOUD);
  float r3 = getOutputScoreRaw(CLASS_SPIKE);
  normalizeCnnOutputs(r0, r1, r2, r3);
}


// ======================= FUSION LOGIC =======================
void updateFusionLogic() {
  bool micBad = (g_micQuality < 35.0f || i2sErrorCount > 1000000UL);
  if (micBad) {
    micErrorMs += DSP_BLOCK_MS;
  } else {
    micErrorMs -= DSP_BLOCK_MS;
    if (micErrorMs < 0) micErrorMs = 0;
  }


  bool spikeCandidate = false;
  if (g_peakDb > SPIKE_MIN_PEAK_DB &&
      g_snrDb > SPIKE_PEAK_ABOVE_NOISE_DB &&
      g_crestDb > SPIKE_MIN_CREST_DB &&
      g_clipPct < 5.0f) {
    spikeCandidate = true;
  }


#if USE_CNN_IN_DECISION
  if (g_cnnSpike > CNN_SPIKE_TH && g_snrDb > 12.0f) spikeCandidate = true;
#endif


  if (spikeCandidate) {
    spikeCandidateMs += DSP_BLOCK_MS;
  } else {
    spikeCandidateMs -= DSP_BLOCK_MS;
    if (spikeCandidateMs < 0) spikeCandidateMs = 0;
  }


  if (spikeCandidateMs >= SPIKE_CONFIRM_MS && millis() > spikeUntilMs) {
    spikeUntilMs = millis() + SPIKE_VISIBLE_MS;
    spikeCount++;
    eventCount++;
    spikeCandidateMs = 0;
  }


  bool loudCandidate = false;
  if (stableState == STATE_LOUD) {
    loudCandidate = (g_spl > LOUD_EXIT_DB && g_snrDb > 8.0f);
  } else {
    loudCandidate = (g_spl > LOUD_ENTER_DB && g_snrDb > 10.0f);
  }


  bool speechCandidate =
    (g_spl >= WARNING_DB_MIN &&
     g_spl <= WARNING_DB_MAX &&
     g_snrDb > 6.0f &&
     g_speechRatio > 0.48f);


  bool prewarningCandidate =
    (g_spl >= WARNING_DB_MIN &&
     g_spl <= WARNING_DB_MAX &&
     g_snrDb > 4.0f &&
     g_speechRatio > 0.38f);


#if USE_CNN_IN_DECISION
  loudCandidate = loudCandidate || (g_cnnLoud > CNN_LOUD_TH && g_snrDb > 8.0f);
  speechCandidate = speechCandidate || (g_cnnSpeech > CNN_SPEECH_TH && g_snrDb > 5.0f);
  prewarningCandidate = prewarningCandidate || (g_cnnSpeech > 0.55f && g_snrDb > 4.0f);
#endif


  bool quietCandidate =
    (g_spl < SMOOTH_DB_MAX &&
     g_snrDb < 5.0f &&
     g_clipPct < 1.0f);


  if (loudCandidate) loudMs += DSP_BLOCK_MS;
  else { loudMs -= DSP_BLOCK_MS; if (loudMs < 0) loudMs = 0; }


  if (speechCandidate) warningMs += DSP_BLOCK_MS;
  else { warningMs -= DSP_BLOCK_MS; if (warningMs < 0) warningMs = 0; }


  if (prewarningCandidate) prewarningMs += DSP_BLOCK_MS;
  else { prewarningMs -= DSP_BLOCK_MS; if (prewarningMs < 0) prewarningMs = 0; }


  if (quietCandidate) quietMs += DSP_BLOCK_MS;
  else quietMs = 0;


  AuraState oldStable = stableState;


  if (micErrorMs >= MIC_ERROR_CONFIRM_MS) {
    stableState = STATE_MIC_ERROR;
  }
  else if (loudMs >= LOUD_CONFIRM_MS) {
    stableState = STATE_LOUD;
  }
  else if (warningMs >= WARNING_CONFIRM_MS) {
    stableState = STATE_WARNING;
  }
  else if (prewarningMs >= PREWARNING_CONFIRM_MS) {
    stableState = STATE_PREWARNING;
  }
  else if (quietCandidate &&
    (stableState == STATE_WARNING || stableState == STATE_LOUD || stableState == STATE_PREWARNING)) {
    stableState = STATE_RECOVERY;
  }


  if (quietMs >= RECOVERY_CONFIRM_MS) {
    stableState = STATE_SMOOTH;
    quietStreakSec++;
  }


  if (oldStable != stableState) eventCount++;


  if (millis() < spikeUntilMs && stableState != STATE_MIC_ERROR) displayState = STATE_SPIKE_OVERLAY;
  else displayState = stableState;
}


// ======================= RECORDING =======================
void appendToRecordingFromBlock() {
  if (!isRecording || recordBuffer == NULL) return;


  const int16_t* src;
#if USE_CLEAN_AUDIO_FOR_RECORD
  src = monitorBlock;
#else
  src = rawBlock;
#endif


  uint32_t remaining = RECORD_SAMPLES - recordWriteSamples;
  uint32_t toCopy = DSP_BLOCK_SAMPLES;
  if (toCopy > remaining) toCopy = remaining;


  memcpy(recordBuffer + recordWriteSamples, src, toCopy * sizeof(int16_t));
  recordWriteSamples += toCopy;


  if (recordWriteSamples >= RECORD_SAMPLES) {
    isRecording = false;
    recordReady = true;
    recordFinishedMs = millis();
    Serial.println("10 second WAV recording finished. Open /record/download to download.");
  }
}


// ======================= WAV HELPERS =======================
void wavSet16(uint8_t* header, int offset, uint16_t value) {
  header[offset + 0] = value & 0xff;
  header[offset + 1] = (value >> 8) & 0xff;
}


void wavSet32(uint8_t* header, int offset, uint32_t value) {
  header[offset + 0] = value & 0xff;
  header[offset + 1] = (value >> 8) & 0xff;
  header[offset + 2] = (value >> 16) & 0xff;
  header[offset + 3] = (value >> 24) & 0xff;
}


void makeWavHeader(uint8_t* header, uint32_t dataBytes) {
  memset(header, 0, 44);
  memcpy(header + 0, "RIFF", 4);
  wavSet32(header, 4, 36 + dataBytes);
  memcpy(header + 8, "WAVE", 4);
  memcpy(header + 12, "fmt ", 4);
  wavSet32(header, 16, 16);
  wavSet16(header, 20, 1);
  wavSet16(header, 22, 1);
  wavSet32(header, 24, SAMPLE_RATE);
  wavSet32(header, 28, SAMPLE_RATE * 2);
  wavSet16(header, 32, 2);
  wavSet16(header, 34, 16);
  memcpy(header + 36, "data", 4);
  wavSet32(header, 40, dataBytes);
}


// ======================= WEB API HANDLERS =======================
void handleAudioInfo() {
  char json[360];
  snprintf(json, sizeof(json),
    "{\"sample_rate\":%d,\"bits_per_sample\":16,\"channels\":1,\"block_samples\":%d,\"block_bytes\":%d,\"record_seconds\":%d,\"live_seq_next\":%lu,\"ring_blocks\":%u}",
    SAMPLE_RATE,
    DSP_BLOCK_SAMPLES,
    (int)(DSP_BLOCK_SAMPLES * sizeof(int16_t)),
    RECORD_SECONDS,
    (unsigned long)liveSeqNext,
    liveRingBlocks
  );
  server.send(200, "application/json", json);
}


void handleAudioBlocks() {
  if (!liveReady || liveRing == NULL || liveRingBlocks == 0) {
    server.send(503, "text/plain", "Live audio not ready");
    return;
  }


  uint32_t latestNext = liveSeqNext;
  uint32_t earliest = 0;
  if (latestNext > liveRingBlocks) earliest = latestNext - liveRingBlocks;


  uint32_t reqSeq;
  if (server.hasArg("seq")) reqSeq = strtoul(server.arg("seq").c_str(), NULL, 10);
  else reqSeq = (latestNext > 0) ? latestNext - 1 : 0;


  uint32_t count = 3;
  if (server.hasArg("count")) count = strtoul(server.arg("count").c_str(), NULL, 10);
  if (count < 1) count = 1;
  if (count > LIVE_FETCH_MAX_BLOCKS) count = LIVE_FETCH_MAX_BLOCKS;


  if (reqSeq < earliest) reqSeq = earliest;


  if (reqSeq >= latestNext) {
    server.sendHeader("X-Audio-Latest-Seq", String(latestNext));
    server.send(204, "text/plain", "");
    return;
  }


  uint32_t available = latestNext - reqSeq;
  if (count > available) count = available;


  WiFiClient client = server.client();
  uint32_t totalBytes = count * DSP_BLOCK_SAMPLES * sizeof(int16_t);


  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("X-Audio-Seq", String(reqSeq));
  server.sendHeader("X-Audio-Blocks", String(count));
  server.setContentLength(totalBytes);
  server.send(200, "application/octet-stream", "");


  for (uint32_t b = 0; b < count && client.connected(); b++) {
    int16_t* ptr = liveBlockPtr(reqSeq + b);
    if (ptr != NULL) client.write((const uint8_t*)ptr, DSP_BLOCK_SAMPLES * sizeof(int16_t));
  }
}


void handleRecordStart() {
  if (isRecording) {
    server.send(200, "application/json", "{\"ok\":true,\"recording\":true,\"message\":\"Recording already running\"}");
    return;
  }


  if (!ensureRecordBuffer()) {
    server.send(500, "application/json", "{\"ok\":false,\"message\":\"Not enough RAM or PSRAM for 10 sec recording buffer\"}");
    return;
  }


  recordWriteSamples = 0;
  recordReady = false;
  isRecording = true;
  recordStartedMs = millis();
  recordFinishedMs = 0;
  memset(recordBuffer, 0, RECORD_BYTES);


  Serial.println("Started 10 second WAV recording.");
  server.send(200, "application/json", "{\"ok\":true,\"recording\":true,\"message\":\"Recording started\"}");
}


void handleRecordStatus() {
  float progress = 0.0f;
  if (RECORD_SAMPLES > 0) progress = (100.0f * recordWriteSamples) / RECORD_SAMPLES;
  if (progress > 100.0f) progress = 100.0f;


  char json[280];
  snprintf(json, sizeof(json),
    "{\"recording\":%s,\"ready\":%s,\"progress\":%.1f,\"samples\":%lu,\"total_samples\":%lu,\"seconds\":%d}",
    isRecording ? "true" : "false",
    recordReady ? "true" : "false",
    progress,
    (unsigned long)recordWriteSamples,
    (unsigned long)RECORD_SAMPLES,
    RECORD_SECONDS
  );
  server.send(200, "application/json", json);
}


void handleRecordDownload() {
  if (!recordReady || recordBuffer == NULL) {
    server.send(409, "text/plain", "Recording is not ready. Click Record first and wait 10 seconds.");
    return;
  }


  uint8_t wavHeader[44];
  makeWavHeader(wavHeader, RECORD_BYTES);


  WiFiClient client = server.client();
  server.sendHeader("Content-Disposition", "attachment; filename=\"aurasync_10sec_clean_pcm.wav\"");
  server.sendHeader("Cache-Control", "no-store");
  server.setContentLength(44 + RECORD_BYTES);
  server.send(200, "audio/wav", "");


  client.write(wavHeader, 44);


  const uint8_t* data = (const uint8_t*)recordBuffer;
  uint32_t bytesLeft = RECORD_BYTES;
  uint32_t offset = 0;


  while (bytesLeft > 0 && client.connected()) {
    uint32_t chunk = bytesLeft;
    if (chunk > 1460) chunk = 1460;
    client.write(data + offset, chunk);
    offset += chunk;
    bytesLeft -= chunk;
    delay(1);
  }
}


void handleCalibrate() {
  if (!server.hasArg("db")) {
    server.send(400, "application/json", "{\"ok\":false,\"message\":\"Use /calibrate?db=65.0 while a sound meter shows 65 dB\"}");
    return;
  }


  float knownDb = server.arg("db").toFloat();
  if (knownDb < 30.0f || knownDb > 120.0f) {
    server.send(400, "application/json", "{\"ok\":false,\"message\":\"Calibration dB must be between 30 and 120\"}");
    return;
  }


  CAL_OFFSET_DB = knownDb - g_rmsDbFs;
  prefs.putFloat("calOffset", CAL_OFFSET_DB);


  char json[180];
  snprintf(json, sizeof(json), "{\"ok\":true,\"known_db\":%.1f,\"rms_dbfs\":%.2f,\"cal_offset\":%.2f}", knownDb, g_rmsDbFs, CAL_OFFSET_DB);
  server.send(200, "application/json", json);
}


void handleNoiseReset() {
  g_noiseDbFs = g_rmsDbFs;
  server.send(200, "application/json", "{\"ok\":true,\"message\":\"Noise floor reset to current room level\"}");
}


void handleApi() {
  char json[1300];


  snprintf(json, sizeof(json),
    "{"
      "\"state\":\"%s\","
      "\"spl\":%.2f,"
      "\"spl_fast\":%.2f,"
      "\"peak\":%.2f,"
      "\"rms_dbfs\":%.2f,"
      "\"peak_dbfs\":%.2f,"
      "\"noise_dbfs\":%.2f,"
      "\"snr_db\":%.2f,"
      "\"crest_db\":%.2f,"
      "\"speech_ratio\":%.3f,"
      "\"clip_pct\":%.2f,"
      "\"zero_pct\":%.2f,"
      "\"dc_offset\":%.1f,"
      "\"mic_quality\":%.1f,"
      "\"cal_offset\":%.2f,"
      "\"cnn_enabled_in_decision\":%s,"
      "\"cnn_ok\":%s,"
      "\"cnn_smooth\":%.3f,"
      "\"cnn_speech\":%.3f,"
      "\"cnn_loud\":%.3f,"
      "\"cnn_spike\":%.3f,"
      "\"event_count\":%d,"
      "\"spike_count\":%d,"
      "\"quiet_streak\":%d,"
      "\"i2s_errors\":%lu,"
      "\"free_heap\":%lu,"
      "\"free_psram\":%lu,"
      "\"live_seq_next\":%lu"
    "}",
    stateName(displayState),
    g_spl,
    g_splFast,
    g_peakDb,
    g_rmsDbFs,
    g_peakDbFs,
    g_noiseDbFs,
    g_snrDb,
    g_crestDb,
    g_speechRatio,
    g_clipPct,
    g_zeroPct,
    g_dcOffset,
    g_micQuality,
    CAL_OFFSET_DB,
#if USE_CNN_IN_DECISION
    "true",
#else
    "false",
#endif
    tinyMlOk ? "true" : "false",
    g_cnnSmooth,
    g_cnnSpeech,
    g_cnnLoud,
    g_cnnSpike,
    eventCount,
    spikeCount,
    quietStreakSec,
    (unsigned long)i2sErrorCount,
    (unsigned long)ESP.getFreeHeap(),
    (unsigned long)ESP.getFreePsram(),
    (unsigned long)liveSeqNext
  );


  server.send(200, "application/json", json);
}


// ======================= WEB DASHBOARD =======================
void handleRoot() {
  String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <title>AuraSync ESP32-S3 N8R8 Final Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { margin:0; font-family:Arial,sans-serif; background:#f3f4f6; color:#222; text-align:center; }
    header { background:#7f0000; color:white; padding:18px; font-size:22px; font-weight:bold; }
    .container { max-width:1050px; margin:20px auto; padding:10px; }
    .stateBox { background:white; padding:24px; border-radius:16px; margin-bottom:18px; box-shadow:0 3px 10px rgba(0,0,0,0.15); }
    #state { font-size:34px; font-weight:bold; color:#7f0000; }
    .sub { color:#666; font-size:13px; margin-top:8px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; }
    .card { background:white; padding:16px; border-radius:14px; box-shadow:0 2px 7px rgba(0,0,0,0.12); }
    .label { color:#666; font-size:13px; }
    .value { font-size:24px; font-weight:bold; margin-top:6px; }
    .good { color:#065f46; } .warn { color:#b45309; } .bad { color:#b00000; }
    .box { background:white; padding:20px; border-radius:16px; margin-top:18px; box-shadow:0 3px 10px rgba(0,0,0,0.15); text-align:left; }
    .boxTitle { font-size:20px; font-weight:bold; color:#7f0000; margin-bottom:10px; text-align:center; }
    .controls { display:flex; flex-wrap:wrap; gap:10px; justify-content:center; align-items:center; margin:14px 0; }
    button,.downloadLink { border:none; border-radius:10px; padding:12px 16px; background:#7f0000; color:white; font-weight:bold; cursor:pointer; text-decoration:none; display:inline-block; }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .downloadLink { background:#065f46; }
    input { padding:11px; border-radius:10px; border:1px solid #ccc; width:110px; }
    .progressOuter { width:100%; height:14px; background:#e5e7eb; border-radius:999px; overflow:hidden; margin-top:8px; }
    .progressInner { height:100%; width:0%; background:#7f0000; transition:width .2s; }
    canvas { width:100%; height:100px; background:#111827; border-radius:12px; margin-top:10px; }
    .small { margin-top:14px; color:#666; font-size:13px; line-height:1.5; text-align:center; }
  </style>
</head>
<body>
<header>AuraSync ESP32-S3 N8R8 Deploy Dashboard</header>
<div class="container">
  <div class="stateBox">
    <div class="label">Current Acoustic State</div>
    <div id="state">Loading...</div>
    <div class="sub" id="statusLine">Waiting for data...</div>
  </div>


  <div class="grid">
    <div class="card"><div class="label">Calibrated SPL Approx.</div><div class="value" id="spl">--</div></div>
    <div class="card"><div class="label">Peak SPL Approx.</div><div class="value" id="peak">--</div></div>
    <div class="card"><div class="label">SNR Above Room</div><div class="value" id="snr">--</div></div>
    <div class="card"><div class="label">Noise Floor dBFS</div><div class="value" id="noise">--</div></div>
    <div class="card"><div class="label">Mic Quality</div><div class="value" id="micQuality">--</div></div>
    <div class="card"><div class="label">Clipping %</div><div class="value" id="clip">--</div></div>
    <div class="card"><div class="label">Zero/Silence %</div><div class="value" id="zero">--</div></div>
    <div class="card"><div class="label">Speech-band Ratio</div><div class="value" id="speechRatio">--</div></div>
    <div class="card"><div class="label">CNN Smooth</div><div class="value" id="cnnSmooth">--</div></div>
    <div class="card"><div class="label">CNN Speech</div><div class="value" id="cnnSpeech">--</div></div>
    <div class="card"><div class="label">CNN Loud</div><div class="value" id="cnnLoud">--</div></div>
    <div class="card"><div class="label">CNN Spike</div><div class="value" id="cnnSpike">--</div></div>
    <div class="card"><div class="label">Event Count</div><div class="value" id="eventCount">--</div></div>
    <div class="card"><div class="label">Spike Count</div><div class="value" id="spikeCount">--</div></div>
    <div class="card"><div class="label">Free PSRAM</div><div class="value" id="psram">--</div></div>
  </div>


  <div class="box">
    <div class="boxTitle">Audio Test: Live Monitor + 10 sec WAV</div>
    <div class="controls">
      <button id="recordBtn" onclick="startRecording()">Record 10 sec WAV</button>
      <a id="downloadBtn" class="downloadLink" href="/record/download" download style="display:none;">Download WAV</a>
      <button id="liveBtn" onclick="toggleLiveAudio()">Start Live Clean Audio</button>
    </div>
    <div class="label" id="recordStatus">Recorder: ready</div>
    <div class="progressOuter"><div class="progressInner" id="recordProgress"></div></div>
    <canvas id="waveCanvas" width="900" height="130"></canvas>
    <div class="small">Live audio uses sequenced 100 ms PCM blocks with browser jitter-buffering. This reduces chirp/clicks caused by skipped or repeated HTTP blocks.</div>
  </div>


  <div class="box">
    <div class="boxTitle">Calibration</div>
    <div class="controls">
      <input id="knownDb" type="number" value="65" min="30" max="120" step="0.1">
      <button onclick="calibrateDb()">Calibrate to Meter dB</button>
      <button onclick="resetNoise()">Reset Room Noise Floor</button>
    </div>
    <div class="small" id="calStatus">Use a real sound-level meter near the mic, type the meter value, then click calibrate. Without this, SPL/dBA is only approximate.</div>
  </div>


  <div class="small">Privacy note: raw/clean audio endpoints are for lab testing. Disable them before public deployment if speech privacy is required.</div>
</div>


<script>
async function updateDashboard() {
  try {
    const r = await fetch('/api?t=' + Date.now());
    const d = await r.json();
    document.getElementById('state').innerText = d.state;
    document.getElementById('spl').innerText = d.spl.toFixed(1);
    document.getElementById('peak').innerText = d.peak.toFixed(1);
    document.getElementById('snr').innerText = d.snr_db.toFixed(1);
    document.getElementById('noise').innerText = d.noise_dbfs.toFixed(1);
    document.getElementById('micQuality').innerText = d.mic_quality.toFixed(0) + '%';
    document.getElementById('clip').innerText = d.clip_pct.toFixed(2);
    document.getElementById('zero').innerText = d.zero_pct.toFixed(1);
    document.getElementById('speechRatio').innerText = d.speech_ratio.toFixed(2);
    document.getElementById('cnnSmooth').innerText = d.cnn_smooth.toFixed(2);
    document.getElementById('cnnSpeech').innerText = d.cnn_speech.toFixed(2);
    document.getElementById('cnnLoud').innerText = d.cnn_loud.toFixed(2);
    document.getElementById('cnnSpike').innerText = d.cnn_spike.toFixed(2);
    document.getElementById('eventCount').innerText = d.event_count;
    document.getElementById('spikeCount').innerText = d.spike_count;
    document.getElementById('psram').innerText = Math.round(d.free_psram / 1024) + ' KB';


    let q = document.getElementById('micQuality');
    q.className = 'value ' + (d.mic_quality > 70 ? 'good' : d.mic_quality > 35 ? 'warn' : 'bad');
    document.getElementById('statusLine').innerText =
      'RMS ' + d.rms_dbfs.toFixed(1) + ' dBFS | Crest ' + d.crest_db.toFixed(1) + ' dB | Cal offset ' + d.cal_offset.toFixed(1) +
      ' | CNN decision: ' + (d.cnn_enabled_in_decision ? 'ON' : 'OFF/safe');
  } catch(e) { console.log(e); }
}
setInterval(updateDashboard, 500); updateDashboard();


async function startRecording() {
  const recordBtn = document.getElementById('recordBtn');
  const downloadBtn = document.getElementById('downloadBtn');
  const status = document.getElementById('recordStatus');
  const bar = document.getElementById('recordProgress');
  recordBtn.disabled = true; downloadBtn.style.display = 'none'; bar.style.width = '0%'; status.innerText = 'Recorder: starting...';
  try {
    const r = await fetch('/record/start?t=' + Date.now());
    const d = await r.json();
    if (!d.ok) { status.innerText = 'Recorder: ' + (d.message || 'failed'); recordBtn.disabled = false; return; }
    pollRecordStatus();
  } catch(e) { status.innerText = 'Recorder error: ' + e; recordBtn.disabled = false; }
}
async function pollRecordStatus() {
  const recordBtn = document.getElementById('recordBtn');
  const downloadBtn = document.getElementById('downloadBtn');
  const status = document.getElementById('recordStatus');
  const bar = document.getElementById('recordProgress');
  try {
    const r = await fetch('/record/status?t=' + Date.now());
    const d = await r.json();
    bar.style.width = d.progress.toFixed(1) + '%';
    if (d.ready) { status.innerText = 'Recorder: complete. WAV is ready.'; recordBtn.disabled = false; downloadBtn.style.display = 'inline-block'; return; }
    if (d.recording) { status.innerText = 'Recorder: ' + d.progress.toFixed(1) + '%'; setTimeout(pollRecordStatus, 250); return; }
    status.innerText = 'Recorder: ready'; recordBtn.disabled = false;
  } catch(e) { status.innerText = 'Recorder status error: ' + e; recordBtn.disabled = false; }
}


let liveAudioOn = false;
let audioCtx = null;
let liveTimer = null;
let nextSeq = 0;
let nextPlayTime = 0;


function drawWaveform(samples) {
  const canvas = document.getElementById('waveCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.beginPath();
  const mid = canvas.height / 2;
  for (let x=0; x<canvas.width; x++) {
    const idx = Math.floor(x * samples.length / canvas.width);
    const y = mid - samples[idx] * (canvas.height * 0.45);
    if (x === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.stroke();
}


async function fetchAudioBlocks() {
  if (!liveAudioOn || !audioCtx) return;
  try {
    const r = await fetch('/audio/blocks?seq=' + nextSeq + '&count=4&t=' + Date.now());
    if (r.status === 204) return;
    if (!r.ok) return;
    const seqHeader = r.headers.get('X-Audio-Seq');
    const blocksHeader = r.headers.get('X-Audio-Blocks');
    const seq = seqHeader ? parseInt(seqHeader) : nextSeq;
    const blocks = blocksHeader ? parseInt(blocksHeader) : 1;
    const ab = await r.arrayBuffer();
    const view = new DataView(ab);
    const sampleCount = Math.floor(ab.byteLength / 2);
    if (sampleCount <= 0) return;


    const audioBuffer = audioCtx.createBuffer(1, sampleCount, 16000);
    const ch = audioBuffer.getChannelData(0);
    for (let i=0; i<sampleCount; i++) ch[i] = view.getInt16(i*2, true) / 32768.0;
    drawWaveform(ch);


    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioCtx.destination);
    const now = audioCtx.currentTime;
    if (nextPlayTime < now + 0.25) nextPlayTime = now + 0.25;
    source.start(nextPlayTime);
    nextPlayTime += audioBuffer.duration;
    nextSeq = seq + blocks;
  } catch(e) { console.log(e); }
}


async function toggleLiveAudio() {
  const btn = document.getElementById('liveBtn');
  if (!liveAudioOn) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    await audioCtx.resume();
    const info = await (await fetch('/audio/info?t=' + Date.now())).json();
    nextSeq = info.live_seq_next;
    nextPlayTime = audioCtx.currentTime + 0.30;
    liveAudioOn = true;
    btn.innerText = 'Stop Live Clean Audio';
    liveTimer = setInterval(fetchAudioBlocks, 90);
    fetchAudioBlocks();
  } else {
    liveAudioOn = false;
    btn.innerText = 'Start Live Clean Audio';
    if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
  }
}


async function calibrateDb() {
  const db = document.getElementById('knownDb').value;
  const s = document.getElementById('calStatus');
  try {
    const r = await fetch('/calibrate?db=' + encodeURIComponent(db) + '&t=' + Date.now());
    const d = await r.json();
    s.innerText = d.ok ? ('Calibrated. New offset: ' + d.cal_offset.toFixed(2) + ' dB') : d.message;
  } catch(e) { s.innerText = 'Calibration failed: ' + e; }
}


async function resetNoise() {
  const s = document.getElementById('calStatus');
  try {
    const r = await fetch('/noise/reset?t=' + Date.now());
    const d = await r.json();
    s.innerText = d.message || 'Noise floor reset.';
  } catch(e) { s.innerText = 'Noise reset failed: ' + e; }
}
</script>
</body>
</html>
)rawliteral";


  server.send(200, "text/html", html);
}


// ======================= SETUP =======================
void setup() {
  Serial.begin(115200);
  delay(1000);


  Serial.println();
  Serial.println("Starting AuraSync ESP32-S3 N8R8 Final Deploy Firmware...");


  prefs.begin("aurasync", false);
  CAL_OFFSET_DB = prefs.getFloat("calOffset", CAL_OFFSET_DB);
  Serial.print("Loaded calibration offset: ");
  Serial.println(CAL_OFFSET_DB);


  allocateLiveRing();
  setupI2S();
  setupTinyML();


  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(ssid, password);


  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }


  Serial.println();
  Serial.print("Dashboard: http://");
  Serial.println(WiFi.localIP());


  if (!MDNS.begin("aurasync")) {
    Serial.println("Error setting up mDNS responder");
  } else {
    Serial.println("Access Dashboard at: http://aurasync.local");
    MDNS.addService("http", "tcp", 80);
  }


  server.on("/", HTTP_GET, handleRoot);
  server.on("/api", HTTP_GET, handleApi);
  server.on("/audio/info", HTTP_GET, handleAudioInfo);
  server.on("/audio/blocks", HTTP_GET, handleAudioBlocks);
  server.on("/record/start", HTTP_GET, handleRecordStart);
  server.on("/record/status", HTTP_GET, handleRecordStatus);
  server.on("/record/download", HTTP_GET, handleRecordDownload);
  server.on("/calibrate", HTTP_GET, handleCalibrate);
  server.on("/noise/reset", HTTP_GET, handleNoiseReset);


  server.begin();
  lastDspMs = millis();
  lastCnnMs = millis();
}


// ======================= MAIN LOOP =======================
void loop() {
  server.handleClient();


  unsigned long now = millis();


  if (now - lastDspMs >= DSP_BLOCK_MS) {
    lastDspMs = now;


    readAudio100msBlock();


#if USE_CLEAN_AUDIO_FOR_LIVE
    pushLiveBlock(monitorBlock);
#else
    pushLiveBlock(rawBlock);
#endif


    appendToRecordingFromBlock();
    computeDSP();
    updateFusionLogic();


    static uint8_t serialDivider = 0;
    serialDivider++;
    if (serialDivider >= 10) {
      serialDivider = 0;
      Serial.print("State: "); Serial.print(stateName(displayState));
      Serial.print(" | SPL: "); Serial.print(g_spl, 1);
      Serial.print(" | Peak: "); Serial.print(g_peakDb, 1);
      Serial.print(" | SNR: "); Serial.print(g_snrDb, 1);
      Serial.print(" | MicQ: "); Serial.print(g_micQuality, 0);
      Serial.print(" | Clip%: "); Serial.print(g_clipPct, 2);
      Serial.print(" | CNN speech: "); Serial.println(g_cnnSpeech, 2);
    }
  }


  if (now - lastCnnMs >= 1000) {
    lastCnnMs = now;
    runCNN();
  }
}





