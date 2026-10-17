#include <math.h>
#include "DHT.h"

#define DHTPIN 2       // Digital pin connected to the DHT sensor
#define DHTTYPE DHT22  // DHT 22 (AM2302), AM2321
DHT dht(DHTPIN, DHTTYPE);

// ─────────────────────────────────────────────────────────────────
// Model Parameters extracted from trained Logistic Regression
// Features (12): Temperature[C], Humidity[%], TVOC[ppb], eCO2[ppm],
//                Raw H2, Raw Ethanol, Pressure[hPa],
//                PM1.0, PM2.5, NC0.5, NC1.0, NC2.5
// Target: Fire Alarm (0=Safe, 1=Alert)
// ─────────────────────────────────────────────────────────────────
const int N_FEATURES = 12;

// StandardScaler Mean (mu)
const float MEAN[N_FEATURES] = {
  31.350938f,    // Temperature[C]
  40.726445f,    // Humidity[%]
  5778.795818f,  // TVOC[ppb]
  5841.181749f,  // eCO2[ppm]
  12781.099073f, // Raw H2
  18997.750114f, // Raw Ethanol
  938.992308f,   // Pressure[hPa]
  2.178146f,     // PM1.0
  3.277121f,     // PM2.5
  18.397425f,    // NC0.5
  7.215046f,     // NC1.0
  1.095907f      // NC2.5
};

// StandardScaler Standard Deviation (sigma)
const float STD[N_FEATURES] = {
  9.210854f,    // Temperature[C]
  16.888109f,   // Humidity[%]
  5496.741514f, // TVOC[ppb]
  6078.783209f, // eCO2[ppm]
  714.349274f,  // Raw H2
  633.993362f,  // Raw Ethanol
  0.727727f,    // Pressure[hPa]
  3.608980f,    // PM1.0
  5.463736f,    // PM2.5
  27.574884f,   // NC0.5
  11.062560f,   // NC1.0
  1.811273f     // NC2.5
};

// Logistic Regression Weights (W)
const float WEIGHTS[N_FEATURES] = {
   3.401914f,  // Temperature[C]
  -1.696985f,  // Humidity[%]
   4.328067f,  // TVOC[ppb]
   3.320843f,  // eCO2[ppm]
  -2.653160f,  // Raw H2
  -1.831437f,  // Raw Ethanol
  -0.071275f,  // Pressure[hPa]
   1.376943f,  // PM1.0
   1.402829f,  // PM2.5
   1.622269f,  // NC0.5
   1.616572f,  // NC1.0
   1.671422f   // NC2.5
};

// Logistic Regression Bias (b)
const float BIAS = 13.062134f;

// ─────────────────────────────────────────────────────────────────
// Default values for features not read from DHT
// (set from first row of dataset as reference)
// ─────────────────────────────────────────────────────────────────
float X[N_FEATURES] = {
  20.0f,      // [0]  Temperature[C]     <- updated by DHT
  57.36f,     // [1]  Humidity[%]        <- updated by DHT
  0.0f,       // [2]  TVOC[ppb]
  400.0f,     // [3]  eCO2[ppm]
  12306.0f,   // [4]  Raw H2
  18520.0f,   // [5]  Raw Ethanol
  939.735f,   // [6]  Pressure[hPa]
  0.0f,       // [7]  PM1.0
  0.0f,       // [8]  PM2.5
  0.0f,       // [9]  NC0.5
  0.0f,       // [10] NC1.0
  0.0f        // [11] NC2.5
};

// ─────────────────────────────────────────────────────────────────
// 1. Standardization: x_scaled = (x_raw - mean) / std
// ─────────────────────────────────────────────────────────────────
float standardize(float x_raw, int idx) {
  return (x_raw - MEAN[idx]) / STD[idx];
}

// ─────────────────────────────────────────────────────────────────
// 2. Sigmoid activation: y = 1 / (1 + e^-z)
// ─────────────────────────────────────────────────────────────────
float sigmoid(float z) {
  return 1.0f / (1.0f + exp(-z));
}

// ─────────────────────────────────────────────────────────────────
// 3. Prediction: z = W . x_scaled + b  ->  sigmoid(z)
// ─────────────────────────────────────────────────────────────────
float predict(float features[]) {
  float z = 0.0f;
  for (int i = 0; i < N_FEATURES; i++) {
    z += WEIGHTS[i] * standardize(features[i], i);
  }
  z += BIAS;
  return sigmoid(z);
}

// ─────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(9600);
  Serial.println(F("=== Logistic Regression - Fire Alarm Detection ==="));
  Serial.println(F("Sensor: DHT22  |  Board: Arduino Uno"));
  Serial.println(F("=================================================="));
  dht.begin();
}

// ─────────────────────────────────────────────────────────────────
void loop() {
  delay(2000);

  // 1. Read DHT22 sensor
  float h = dht.readHumidity();
  float t = dht.readTemperature();

  // 2. Validate sensor readings
  if (isnan(h) || isnan(t)) {
    Serial.println(F("[ERROR] Failed to read from DHT sensor!"));
    return;
  }

  // 3. Update input array with live sensor values
  X[0] = t;  // Temperature[C]
  X[1] = h;  // Humidity[%]

  // 4. Compute prediction
  float y_pred = predict(X);
  int   y_class = (y_pred >= 0.5f) ? 1 : 0;

  // 5. Print results to Serial Monitor
  Serial.println(F("--------------------------------------------------"));
  Serial.print(F("Temperature : ")); Serial.print(t, 2); Serial.println(F(" C"));
  Serial.print(F("Humidity    : ")); Serial.print(h, 2); Serial.println(F(" %"));
  Serial.print(F("Probability : ")); Serial.println(y_pred, 4);
  Serial.print(F("Prediction  : Class ")); Serial.print(y_class);

  if (y_class == 1) {
    Serial.println(F("  -->  *** FIRE ALARM! ***"));
  } else {
    Serial.println(F("  -->  Safe (Normal)"));
  }
}