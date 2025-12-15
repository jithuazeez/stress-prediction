"""
PPG (Photoplethysmogram) signal processing.

Based on Iqbal et al. (2022) "Stress Monitoring Using Wearable Sensors" methodology.
Implements BVP extraction and peak detection for HR/HRV calculation.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
from typing import Tuple, List, Optional


class PPGProcessor:
    """
    Process PPG signals to extract BVP (Blood Volume Pulse) and inter-beat intervals.
    
    Following Stress-Predict paper methodology:
    - High-pass filter (0.05-0.5 Hz) to obtain BVP from raw PPG
    - Detect systolic peaks (local maxima) for heart beats
    - Calculate inter-beat intervals (IBI) from peaks
    """
    
    def __init__(self, sampling_rate: float = 64.0, 
                 highpass_cutoff: float = 0.05,
                 lowpass_cutoff: float = 8.0,
                 filter_order: int = 4):
        """
        Initialize PPG processor.
        
        Args:
            sampling_rate: PPG sampling rate in Hz (Empatica E4 uses 64 Hz)
            highpass_cutoff: High-pass filter cutoff (Hz) to remove baseline drift
            lowpass_cutoff: Low-pass filter cutoff (Hz) to remove high-frequency noise
            filter_order: Butterworth filter order
        """
        self.sampling_rate = sampling_rate
        self.highpass_cutoff = highpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff
        self.filter_order = filter_order
    
    def extract_bvp(self, ppg_signal: np.ndarray) -> np.ndarray:
        """
        Extract Blood Volume Pulse (BVP) from raw PPG signal.
        
        Applies bandpass filtering to remove DC component and noise.
        
        Args:
            ppg_signal: Raw PPG signal values
        
        Returns:
            Filtered BVP signal
        """
        if len(ppg_signal) < 100:
            return ppg_signal
        
        # Design bandpass filter
        nyquist = 0.5 * self.sampling_rate
        low = self.highpass_cutoff / nyquist
        high = self.lowpass_cutoff / nyquist
        
        # Ensure filter parameters are valid
        if low >= 1.0:
            low = 0.01
        if high >= 1.0:
            high = 0.99
        
        try:
            b, a = butter(self.filter_order, [low, high], btype='band')
            bvp = filtfilt(b, a, ppg_signal)
            return bvp
        except Exception as e:
            print(f"Warning: BVP extraction failed: {e}")
            return ppg_signal
    
    def detect_peaks(self, bvp_signal: np.ndarray, 
                    min_distance_seconds: float = 0.4) -> np.ndarray:
        """
        Detect systolic peaks in BVP signal.
        
        Peaks represent heart beats. min_distance ensures physiologically valid detection
        (0.4s = max 150 BPM, typical upper limit).
        
        Args:
            bvp_signal: Filtered BVP signal
            min_distance_seconds: Minimum time between peaks in seconds
        
        Returns:
            Array of peak indices
        """
        if len(bvp_signal) < 10:
            return np.array([])
        
        # Calculate minimum distance in samples
        min_distance_samples = int(min_distance_seconds * self.sampling_rate)
        
        # Detect peaks with adaptive threshold
        threshold = 0.3 * np.max(np.abs(bvp_signal))
        prominence = 0.2 * (np.max(bvp_signal) - np.min(bvp_signal))
        
        peaks, properties = find_peaks(bvp_signal,
                                      distance=min_distance_samples,
                                      height=threshold,
                                      prominence=prominence)
        
        return peaks
    
    def calculate_ibi(self, peaks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Inter-Beat Intervals (IBI) from detected peaks.
        
        IBI is the time difference between consecutive heart beats in milliseconds.
        
        Args:
            peaks: Array of peak indices
        
        Returns:
            Tuple of (ibi_times, ibi_values)
            - ibi_times: Time of each IBI in seconds
            - ibi_values: IBI durations in milliseconds
        """
        if len(peaks) < 2:
            return np.array([]), np.array([])
        
        # Calculate time differences between peaks
        peak_times = peaks / self.sampling_rate  # Convert to seconds
        ibi_values = np.diff(peak_times) * 1000  # Convert to milliseconds
        ibi_times = peak_times[1:]  # Time of each IBI (at the second peak)
        
        # Filter physiologically valid IBIs (300-2000 ms = 30-200 BPM)
        valid_mask = (ibi_values >= 300) & (ibi_values <= 2000)
        
        return ibi_times[valid_mask], ibi_values[valid_mask]
    
    def process_ppg_window(self, ppg_signal: np.ndarray) -> dict:
        """
        Process PPG signal window to extract features.
        
        Args:
            ppg_signal: Raw PPG values for a window
        
        Returns:
            Dictionary of PPG features
        """
        features = {}
        
        # Extract BVP
        bvp = self.extract_bvp(ppg_signal)
        
        # BVP statistics
        features['bvp_mean'] = np.mean(bvp)
        features['bvp_std'] = np.std(bvp)
        features['bvp_min'] = np.min(bvp)
        features['bvp_max'] = np.max(bvp)
        features['bvp_range'] = features['bvp_max'] - features['bvp_min']
        
        # Detect peaks
        peaks = self.detect_peaks(bvp)
        
        if len(peaks) > 0:
            # Peak count
            features['num_peaks'] = len(peaks)
            
            # Peak amplitudes
            peak_amplitudes = bvp[peaks]
            features['peak_amplitude_mean'] = np.mean(peak_amplitudes)
            features['peak_amplitude_std'] = np.std(peak_amplitudes)
            
            # Calculate IBIs
            ibi_times, ibi_values = self.calculate_ibi(peaks)
            
            if len(ibi_values) > 0:
                # Instantaneous heart rate from IBIs
                hr_values = 60000 / ibi_values  # Convert to BPM
                features['hr_from_ppg_mean'] = np.mean(hr_values)
                features['hr_from_ppg_std'] = np.std(hr_values)
                
                # Signal quality indicator
                features['signal_quality'] = len(ibi_values) / len(peaks) if len(peaks) > 0 else 0
            else:
                features['hr_from_ppg_mean'] = np.nan
                features['hr_from_ppg_std'] = np.nan
                features['signal_quality'] = 0
        else:
            # No peaks detected - poor signal quality
            features['num_peaks'] = 0
            features['peak_amplitude_mean'] = np.nan
            features['peak_amplitude_std'] = np.nan
            features['hr_from_ppg_mean'] = np.nan
            features['hr_from_ppg_std'] = np.nan
            features['signal_quality'] = 0
        
        return features


def extract_ppg_features(ppg_data: pd.DataFrame, 
                        value_column: str = 'value',
                        sampling_rate: float = 64.0) -> dict:
    """
    Extract PPG features from window data.
    
    Args:
        ppg_data: DataFrame with PPG values
        value_column: Name of column containing PPG values
        sampling_rate: PPG sampling rate in Hz
    
    Returns:
        Dictionary of extracted features
    """
    if value_column not in ppg_data.columns:
        print(f"Warning: {value_column} not found in PPG data")
        return {}
    
    ppg_signal = ppg_data[value_column].values
    
    processor = PPGProcessor(sampling_rate=sampling_rate)
    features = processor.process_ppg_window(ppg_signal)
    
    return features


if __name__ == '__main__':
    # Test PPG processing
    print("Testing PPG processor...")
    
    # Simulate PPG signal (simple sinusoid with noise)
    t = np.linspace(0, 10, 640)  # 10 seconds at 64 Hz
    ppg_clean = np.sin(2 * np.pi * 1.2 * t)  # 72 BPM
    ppg_noisy = ppg_clean + 0.1 * np.random.randn(len(t))
    
    processor = PPGProcessor()
    
    # Extract BVP
    bvp = processor.extract_bvp(ppg_noisy)
    print(f"✓ BVP extracted, length: {len(bvp)}")
    
    # Detect peaks
    peaks = processor.detect_peaks(bvp)
    print(f"✓ Peaks detected: {len(peaks)}")
    
    # Calculate IBI
    ibi_times, ibi_values = processor.calculate_ibi(peaks)
    print(f"✓ IBIs calculated: {len(ibi_values)}")
    if len(ibi_values) > 0:
        hr = 60000 / np.mean(ibi_values)
        print(f"  Mean HR: {hr:.1f} BPM")
    
    # Extract features
    features = processor.process_ppg_window(ppg_noisy)
    print(f"\n✓ Features extracted: {len(features)} features")
    for key, value in features.items():
        print(f"  {key}: {value:.2f}" if not np.isnan(value) else f"  {key}: NaN")
    
    print("\n✅ PPG processing working!")








