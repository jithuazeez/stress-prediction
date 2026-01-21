# Dataset Description: VitaStress

This document describes the VitaStress dataset, a multimodal wearable sensor dataset for stress recognition research.

---

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| **Dataset Name** | VitaStress |
| **Reference** | Schreiber et al. (2025), "Stress Detection from Multimodal Wearable Sensor Data" |
| **Availability** | Public (GitHub: paulvincenz/VitaStress) |
| **Collection Context** | Controlled laboratory setting |
| **Total Participants** | 21 subjects |
| **Demographics** | 18 male, 3 female; Mean age: 24.1 years (SD: 3.97) |
| **Total Recording Duration** | ~1,072 minutes (~18 hours) |
| **Stimuli Duration** | ~542.5 minutes (~9 hours) |
| **Wearable Device** | Corsano Cardiowatch 287-2B (EU CE MDR certified, FDA cleared) |

---

## 2. Data Files Per Subject

Each subject folder contains 13 data files. The directory structure is:

```
VitaStress/data/
└── id_<subject_uuid>/
    ├── <uuid>_acc.csv
    ├── <uuid>_activity.csv
    ├── <uuid>_annotation.csv
    ├── <uuid>_bioz.csv
    ├── <uuid>_emography.csv
    ├── <uuid>_heat_flux_sensor_temperature.csv
    ├── <uuid>_ppg2_green_6.csv
    ├── <uuid>_ppg2_infra_red_22.csv
    ├── <uuid>_ppg2_red_182.csv
    ├── <uuid>_ppg3.csv
    ├── <uuid>_rr_interval.csv
    ├── <uuid>_temperature.csv
    └── <uuid>_selfreport_.csv
```

---

## 3. Raw Signal Files

### 3.1 Accelerometer (`acc.csv`)

| Property | Value |
|----------|-------|
| **Sampling Rate** | ~32 Hz (31 ms intervals) |
| **Samples per Subject** | ~146,000 |
| **Missing Data** | <1% |

**Columns:**

| Column | Description | Unit/Range |
|--------|-------------|------------|
| `date` | Timestamp (ISO 8601) | UTC |
| `metric_id` | Sensor identifier | Hex (0x2b) |
| `chunk_index` | Data chunk number | Integer |
| `quality` | Signal quality flag | 0-5 (higher = better) |
| `body_pose` | Body position indicator | Integer |
| `accX` | Acceleration X-axis | Raw ADC units (milli-g) |
| `accY` | Acceleration Y-axis | Raw ADC units (milli-g) |
| `accZ` | Acceleration Z-axis | Raw ADC units (milli-g) |

**Example:**
```csv
date,metric_id,chunk_index,quality,body_pose,accX,accY,accZ
2035-03-15 14:50:00+00:00,0x2b,156,4,1,30,-26,504
2035-03-15 14:50:00.031000+00:00,0x2b,156,4,1,30,-26,504
```

---

### 3.2 PPG Green LED (`ppg2_green_6.csv`)

| Property | Value |
|----------|-------|
| **Sampling Rate** | ~64 Hz (15.6 ms intervals) |
| **Samples per Subject** | ~134,000 |
| **Missing Data** | <1% |

**Columns:**

| Column | Description | Unit/Range |
|--------|-------------|------------|
| `date` | Timestamp (ISO 8601) | UTC |
| `metric_id` | Sensor identifier | Hex (0x7e) |
| `chunk_index` | Data chunk number | Integer |
| `quality` | Signal quality flag | 0-5 |
| `body_pose` | Body position indicator | Integer |
| `led_pd_pos` | LED/photodiode position | Integer (6) |
| `offset` | Signal offset | Integer |
| `exp` | Exposure setting | Integer |
| `led` | LED intensity | Integer |
| `gain` | Amplifier gain | Integer |
| `value` | Raw PPG signal | Raw ADC units |

**Example:**
```csv
date,metric_id,chunk_index,quality,body_pose,led_pd_pos,offset,exp,led,gain,value
2035-03-15 14:50:00+00:00,0x7e,132,4,1,6,0,0,34,2,32175
2035-03-15 14:50:00.015000+00:00,0x7e,132,4,1,6,0,0,34,2,32174
```

---

### 3.3 PPG Infrared LED (`ppg2_infra_red_22.csv`)

| Property | Value |
|----------|-------|
| **Sampling Rate** | ~64 Hz (15.6 ms intervals) |
| **Samples per Subject** | ~132,000 |
| **Missing Data** | <1% |

**Columns:** Same structure as `ppg2_green_6.csv` with `led_pd_pos=22` and `metric_id=0x7b`.

---

### 3.4 PPG Red LED (`ppg2_red_182.csv`)

| Property | Value |
|----------|-------|
| **Sampling Rate** | ~64 Hz (15.6 ms intervals) |
| **Samples per Subject** | ~130,000 |
| **Missing Data** | <1% |

**Columns:** Same structure as `ppg2_green_6.csv` with `led_pd_pos=182` and `metric_id=0x7c`.

---

### 3.5 PPG Alternate Format (`ppg3.csv`)

| Property | Value |
|----------|-------|
| **Sampling Rate** | ~32 Hz (31 ms intervals) |
| **Samples per Subject** | ~121,000 |
| **Missing Data** | <1% |

**Columns:**

| Column | Description | Unit/Range |
|--------|-------------|------------|
| `date` | Timestamp (ISO 8601) | UTC |
| `ppg` | Raw PPG signal | Raw ADC units |
| `acc` | Accelerometer quality flag | 0/1 |
| `gain` | Amplifier gain | Integer |
| `led` | LED intensity | Integer |
| `acc_x` | Acceleration X-axis | Raw ADC units |
| `acc_y` | Acceleration Y-axis | Raw ADC units |
| `acc_z` | Acceleration Z-axis | Raw ADC units |
| `crc` | Checksum | Integer |

**Example:**
```csv
date,ppg,acc,gain,led,acc_x,acc_y,acc_z,crc
2035-03-15 14:50:00.002000+00:00,48812,1,2,34,28,-28,500,489165101
```

---

### 3.6 Bioimpedance (`bioz.csv`)

| Property | Value |
|----------|-------|
| **Sampling Rate** | ~25 Hz (40 ms intervals) |
| **Samples per Subject** | ~133,000 |
| **Missing Data** | <1% |

**Columns:**

| Column | Description | Unit/Range |
|--------|-------------|------------|
| `date` | Timestamp (ISO 8601) | UTC |
| `metric_id` | Sensor identifier | Hex (0x3d) |
| `chunk_index` | Data chunk number | Integer |
| `value` | Bioimpedance signal | Raw ADC units |

**Example:**
```csv
date,metric_id,chunk_index,value
2035-03-15 14:50:00+00:00,0x3d,114,1112
2035-03-15 14:50:00.040000+00:00,0x3d,114,1112
```

---

### 3.7 Heat Flux Sensor Temperature (`heat_flux_sensor_temperature.csv`)

| Property | Value |
|----------|-------|
| **Sampling Rate** | 1 Hz (1 second intervals) |
| **Samples per Subject** | ~5,400 |
| **Missing Data** | <1% |

**Columns:**

| Column | Description | Unit/Range |
|--------|-------------|------------|
| `date` | Timestamp (ISO 8601) | UTC |
| `skin_temp` | Skin temperature | °C (typical: 20-40) |
| `heatflux` | Heat flux | W/m² (typical: -50 to 200) |
| `acc_x` | Low-rate acceleration X | Normalised (-1 to 1) |
| `acc_y` | Low-rate acceleration Y | Normalised (-1 to 1) |
| `acc_z` | Low-rate acceleration Z | Normalised (-1 to 1) |
| `pulse_rate` | Device-derived heart rate | BPM |
| `cbt` | Core body temperature estimate | °C (typical: 35-40) |

**Example:**
```csv
date,skin_temp,heatflux,acc_x,acc_y,acc_z,pulse_rate,cbt
2035-03-15 14:50:00+00:00,19.99,18.58,-0.04,-0.04,-0.98,59,36.96
2035-03-15 14:50:01+00:00,19.95,18.58,-0.04,-0.04,-0.98,60,36.96
```

---

### 3.8 Temperature (`temperature.csv`)

| Property | Value |
|----------|-------|
| **Sampling Rate** | ~0.033 Hz (30 second intervals) |
| **Samples per Subject** | ~180 |
| **Missing Data** | <5% for temp_sk1; temp_sk2 and temp_amb often unavailable |

**Columns:**

| Column | Description | Unit/Range |
|--------|-------------|------------|
| `date` | Timestamp (ISO 8601) | UTC |
| `temp_sk1` | Skin temperature (sensor 1) | °C |
| `temp_sk2` | Skin temperature (sensor 2) | °C (often 0/missing) |
| `temp_amb` | Ambient temperature | °C (often missing) |

**Example:**
```csv
date,temp_sk1,temp_sk2,temp_amb
2035-03-15 14:50:00+00:00,36.96,0,
2035-03-15 14:50:30+00:00,36.96,0,
```

**Note:** `temp_sk2` and `temp_amb` are frequently zero or empty across subjects.

---

### 3.9 Emography / Electrodermal Activity (`emography.csv`)

| Property | Value |
|----------|-------|
| **Sampling Rate** | ~0.017 Hz (60 second intervals) |
| **Samples per Subject** | ~90 |
| **Missing Data** | <5% |

**Columns:**

| Column | Description | Unit/Range |
|--------|-------------|------------|
| `date` | Timestamp (ISO 8601) | UTC |
| `cz` | Unknown metric | Integer |
| `pcz` | Unknown metric | Integer |
| `pczt` | Unknown metric | Integer |
| `czh` | Unknown metric | Integer |
| `cc` | Contact capacitance | Integer |
| `quality` | Signal quality flag | 0-5 |
| `stress_skin` | Skin conductance (EDA) | Arbitrary units |
| `stress_skin_quality` | EDA quality flag | 0-5 |

**Example:**
```csv
date,cz,pcz,pczt,czh,cc,quality,stress_skin,stress_skin_quality
2035-03-15 14:50:35+00:00,1,1,255,1,0,4,0,4
2035-03-15 15:09:37+00:00,1,1,255,1,0,4,281,4
```

**Note:** The extremely low sampling rate (~1 sample/minute) limits the utility of this signal for window-based analysis.

---

### 3.10 RR Interval (`rr_interval.csv`)

| Property | Value |
|----------|-------|
| **Sampling Rate** | Event-based (non-uniform) |
| **Samples per Subject** | ~3,400 (highly variable) |
| **Missing Data** | >50% overall; >85% during physical activity |

**Columns:**

| Column | Description | Unit/Range |
|--------|-------------|------------|
| `date` | Timestamp (ISO 8601) | UTC |
| `rr` | RR interval (time between heartbeats) | Milliseconds |

**Example:**
```csv
date,rr
2035-03-15 14:59:22+00:00,529
2035-03-15 14:59:25+00:00,515
2035-03-15 15:01:13+00:00,701
```

**Known Issues:**
- Highly irregular timestamps (gaps of seconds to minutes)
- Significant data loss during physical activity (motion artifacts)
- Not reliable for HRV computation in this dataset

---

### 3.11 Activity (`activity.csv`)

| Property | Value |
|----------|-------|
| **Sampling Rate** | ~0.033 Hz (30 second intervals) |
| **Samples per Subject** | ~180 |
| **Missing Data** | Variable by metric (see below) |

**Columns:**

| Column | Description | Unit/Range | Missing % |
|--------|-------------|------------|-----------|
| `date` | Timestamp | UTC | 0% |
| `bpm` | Heart rate | BPM | Variable |
| `bpm_q` | HR quality flag | 0-5 | Variable |
| `last_steps` | Step count | Integer | Low |
| `activity_type` | Activity classification | Integer | Low |
| `speed` | Movement speed | m/s | Low |
| `skin_proximity` | Skin contact indicator | Integer | Low |
| `energy_exp` | Energy expenditure | Arbitrary | Low |
| `respiration_rate` | Breathing rate | Breaths/min | ~40-50% |
| `spo2` | Oxygen saturation | % (0-100) | Variable |
| `spo2_q` | SpO2 quality flag | 0-5 | Variable |
| `wearing` | Wearing status | Integer | Low |
| `battery` | Battery level | % | 0% |
| ... | (additional metrics) | ... | ... |

**Example:**
```csv
date,bpm,bpm_q,last_steps,activity_type,speed,skin_proximity,energy_exp,respiration_rate,...
2035-03-15 14:50:00+00:00,120,1,0,7,0,0,1757,0.0,...
2035-03-15 15:02:01+00:00,84,1,0,7,0,0,631,12.0,...
```

**Note:** Many `bpm` values are 0 (device not detecting HR), and `respiration_rate` has high missingness.

---

## 4. Annotation and Self-Report Files

### 4.1 Annotation (`annotation.csv`)

| Property | Value |
|----------|-------|
| **Records per Subject** | ~40-60 |
| **Missing Data** | 0% |

**Columns:**

| Column | Description |
|--------|-------------|
| `timestamp` | Event timestamp (ISO 8601) |
| `Button Name` | Event label |

**Event Types:**

| Event Label | Description |
|-------------|-------------|
| `Baseline Start (Start of Experiment)` | Beginning of neutral baseline |
| `Baseline Stop` | End of neutral baseline |
| `Cognitive: Introduction` | Cognitive task explanation |
| `Cognitive: Start` | Beginning of cognitive stress (mental arithmetic) |
| `Cognitive: Mistake` | Subject made an error |
| `Cognitive: Stop` | End of cognitive stress |
| `Physical: Introduction` | Physical task explanation |
| `Physical: Start` | Beginning of physical activity |
| `Physical: Warmup` | Warm-up phase |
| `Physical: Intensity increase` | Speed/intensity increased |
| `Physical: Stop` | End of physical activity |
| `Public Speaking: Introduction` | Speech task explanation |
| `Public Speaking: Preparation Start` | Subject begins preparing speech |
| `Public Speaking: Preparation Stop` | End of preparation |
| `Public Speaking Start` | Subject begins delivering speech |
| `Public Speaking Stop` | End of speech |
| `Rest: Start` | Beginning of rest period |
| `Rest: Stop` | End of rest period |
| `Self-Report: Start` | Subject filling questionnaire |
| `Self-Report: Stop` | End of questionnaire |
| `Sitting`, `Standing`, `Walking` | Body posture markers |
| `Comment: <text>` | Experimenter notes |

**Example:**
```csv
timestamp,Button Name
2035-03-15 15:12:28.061472+00:00,Baseline Start (Start of Experiment)
2035-03-15 15:22:29.092427+00:00,Baseline Stop
2035-03-15 15:26:09.148608+00:00,Cognitive: Start
2035-03-15 15:31:09.653759+00:00,Cognitive: Stop
2035-03-15 15:39:59.855568+00:00,Physical: Start
```

---

### 4.2 Self-Report (`selfreport_.csv`)

| Property | Value |
|----------|-------|
| **Records per Subject** | 4 (one per condition) |
| **Missing Data** | 0% |

**Columns:**

| Column | Description | Range |
|--------|-------------|-------|
| (index) | Condition | rest, physical, cognitive, social |
| `arousal` | Self-reported arousal | 1-9 (1=low, 9=high) |
| `pleasure` | Self-reported valence | 1-9 (1=negative, 9=positive) |
| `dominance` | Self-reported control | 1-9 (1=low, 9=high) |

**Example:**
```csv
,arousal,pleasure,dominance
rest,1,9,3
physical,1,9,9
cognitive,8,1,1
social,2,4,7
```

---

## 5. Sampling Rate Summary

| File | Signal Type | Sampling Rate | Samples/Subject |
|------|-------------|---------------|-----------------|
| `acc.csv` | Accelerometer | ~32 Hz | ~146,000 |
| `ppg2_green_6.csv` | PPG (Green LED) | ~64 Hz | ~134,000 |
| `ppg2_infra_red_22.csv` | PPG (IR LED) | ~64 Hz | ~132,000 |
| `ppg2_red_182.csv` | PPG (Red LED) | ~64 Hz | ~130,000 |
| `ppg3.csv` | PPG (Alternate) | ~32 Hz | ~121,000 |
| `bioz.csv` | Bioimpedance | ~25 Hz | ~133,000 |
| `heat_flux_sensor_temperature.csv` | Thermal | 1 Hz | ~5,400 |
| `temperature.csv` | Temperature | ~0.033 Hz | ~180 |
| `emography.csv` | EDA | ~0.017 Hz | ~90 |
| `activity.csv` | Device metrics | ~0.033 Hz | ~180 |
| `rr_interval.csv` | RR intervals | Event-based | ~3,400 |
| `annotation.csv` | Events | Event-based | ~50 |
| `selfreport_.csv` | Questionnaire | Per-condition | 4 |

---

## 6. Data Quality Analysis

### 6.1 Missing Data Summary

| Signal | Source File | Mean Missing % | Quality |
|--------|-------------|----------------|---------|
| Accelerometer (X, Y, Z) | `acc.csv` | <1% | Excellent |
| PPG Green | `ppg2_green_6.csv` | <1% | Excellent |
| PPG Infrared | `ppg2_infra_red_22.csv` | <1% | Excellent |
| PPG Red | `ppg2_red_182.csv` | <1% | Excellent |
| Bioimpedance | `bioz.csv` | <1% | Excellent |
| Skin Temperature | `heat_flux_sensor_temperature.csv` | <1% | Excellent |
| Heat Flux | `heat_flux_sensor_temperature.csv` | <1% | Excellent |
| Core Body Temp | `heat_flux_sensor_temperature.csv` | <1% | Excellent |
| temp_sk1 | `temperature.csv` | <5% | Good |
| temp_sk2 | `temperature.csv` | >90% (zeros) | Poor |
| temp_amb | `temperature.csv` | >95% (missing) | Poor |
| EDA (stress_skin) | `emography.csv` | <5% | Moderate (low rate) |
| HR (bpm) | `activity.csv` | Variable | Moderate |
| Respiration Rate | `activity.csv` | ~40-50% | Poor |
| RR Intervals | `rr_interval.csv` | >50% overall, >85% during movement | Poor |

### 6.2 Quality Recommendations

| Signal | Recommendation | Reason |
|--------|----------------|--------|
| Accelerometer | **Use** | Near-complete, high sampling rate |
| PPG Green | **Use** | Near-complete, best for HR extraction |
| skin_temp (heat flux) | **Use** | Near-complete, 1 Hz rate |
| heatflux | **Use** | Near-complete, 1 Hz rate |
| cbt | **Use** | Near-complete, 1 Hz rate |
| Bioimpedance | **Consider** | Near-complete, novel signal |
| PPG IR/Red | **Consider** | Useful for SpO2 if needed |
| temp_sk1 | **Alternative** | Lower rate than heat flux sensor |
| EDA | **Exclude** | Sampling rate too low (~1/min) |
| RR intervals | **Exclude** | Too much missing data, unreliable |
| Respiration rate | **Exclude** | ~50% missing |
| temp_sk2, temp_amb | **Exclude** | Mostly zeros/missing |

---

## 7. Stress Protocol

### 7.1 Experimental Phases

| Phase | Description | Duration |
|-------|-------------|----------|
| **Neutral Baseline** | Seated, calming music, noise-canceling headphones | ~10 min |
| **Cognitive Stress** | Mental arithmetic (counting down by 13); restart on error | ~5 min |
| **Rest Period** | Seated recovery | ~5 min |
| **Physical Stress** | Treadmill running at 3 incremental speeds | ~6 min |
| **Rest Period** | Seated recovery | ~5 min |
| **Socio-Evaluative Stress** | Public speaking preparation + delivery (TSST-inspired) | ~10 min |
| **Recovery** | Seated, calming music | ~5 min |

### 7.2 Stress Types

| Stress Type | Annotation Events | Characteristics |
|-------------|-------------------|-----------------|
| **Cognitive** | `Cognitive: Start` to `Cognitive: Stop` | Mental arithmetic, standing |
| **Physical** | `Physical: Start` to `Physical: Stop` | Treadmill exercise, elevated HR |
| **Socio-Evaluative** | `Public Speaking: Preparation Start` to `Public Speaking Stop` | Speech task, social anxiety |

---

## 8. References

1. Schreiber, P., Cinar, B., Mackert, L., & Maleshkova, M. (2025). *Stress Detection from Multimodal Wearable Sensor Data*. arXiv:2508.10468v1.

2. Schreiber, P., Grensing, F., & Maleshkova, M. (2024). *TRRRACED–Towards Reproducible, Replicable and Reusable Affective Computing Experiments and Data*. IEEE CogInfoCom.

3. Corsano Health. *Corsano Cardiowatch 287-2B Technical Documentation*. https://corsano.com/

4. Kirschbaum, C., Pirke, K. M., & Hellhammer, D. H. (1993). *The 'Trier Social Stress Test'–a Tool for Investigating Psychobiological Stress Responses in a Laboratory Setting*. Neuropsychobiology.
