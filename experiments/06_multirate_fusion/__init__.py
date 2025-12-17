"""
Experiment 06: Multi-Rate Late Fusion.

Processes signals at their native sampling rates:
- PPG: 64 Hz (raw photoplethysmography)
- ACC: 32 Hz (accelerometer)
- Temp: 1 Hz (skin temperature)

Uses separate encoders per modality with late fusion of embeddings.
"""
