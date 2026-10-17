# 🚨 AIoT: Fire Alarm Detection System (TinyML & Edge AI)

An end-to-end IoT and Edge AI project implementing **Fire Alarm Detection** using **Logistic Regression** deployed directly on an **Arduino Uno (ATmega328P)** microcontroller with a **DHT22** sensor in the **Wokwi** simulation environment.

---

## 📌 Project Overview

This repository encompasses the full development cycle:
1. **TP1 - Exploratory Data Analysis & Baseline**: Feature analysis and initial classification modeling on the Smoke Detection IoT dataset.
2. **TP2 - Model Training & Efficiency Analysis**: Building machine learning pipelines (`StandardScaler` + `LogisticRegression`), model footprint, and inference latency benchmarking.
3. **TP3 - Edge Deployment on Microcontroller**: Porting the mathematical components (Standardization, Dot Product, Sigmoid activation) to C++/Arduino and deploying in real time.

---

## 🏗️ Hardware Architecture & Circuit

- **MCU**: Arduino Uno (ATmega328P, 16MHz, 2KB RAM, 32KB Flash)
- **Sensor**: DHT22 (AM2302) Temperature & Humidity Sensor
- **Pinout**:
  - `VCC` ➔ Arduino `5V`
  - `DATA / SDA` ➔ Arduino Digital Pin `2`
  - `GND` ➔ Arduino `GND`

---

## 🧠 TinyML Edge Implementation

Because microcontrollers have limited RAM and lack Python ML libraries, the inference pipeline is implemented from scratch in pure C++:

### 1. Standardization:
$$x_{\text{scaled}} = \frac{x_{\text{raw}} - \mu}{\sigma}$$

### 2. Linear Combination:
$$z = \sum_{i=0}^{N-1} (W_i \cdot x_{\text{scaled}, i}) + b$$

### 3. Sigmoid Activation:
$$\hat{y} = \frac{1}{1 + e^{-z}}$$

- **Decision Threshold**: $\hat{y} \ge 0.5 \implies$ **Fire Alarm (Class 1)**, otherwise **Safe (Class 0)**.

---

## 📁 Repository Structure

```text
├── TP1_Fire_Alarm_Detection_FINAL.ipynb   # TP1 Analysis & Baseline Notebook
├── notebookb027616146.ipynb               # TP2 Pipelines & Efficiency Benchmarks
├── TP3/                                  # Embedded Deployment Project
│   ├── platformio.ini                    # PlatformIO Configuration
│   ├── diagram.json                      # Wokwi Circuit Diagram
│   ├── wokwi.toml                        # Wokwi Simulator Configuration
│   ├── TP3_Logistic_Arduino.md           # Assignment Instructions
│   └── src/
│       └── main.cpp                      # Arduino C++ Firmware (Standardize + Predict)
├── .gitignore
└── README.md
```

---

## ⚡ How to Build & Run

### Using PlatformIO (CLI):
```bash
cd TP3
pio run
```

### Running Simulation with Wokwi:
1. Open the project in **VS Code** with the **Wokwi** extension installed.
2. Open `TP3/diagram.json`.
3. Press `Ctrl + Shift + P` and select `Wokwi: Start Simulator` (or click the green Play ▶ button).
4. Interact with the DHT22 sensor to observe real-time predictions in the Serial Monitor (`9600 baud`).

---

## 👤 Author
- **Yacine Bellou**
- Course: AI for IoT / Embedded AI
