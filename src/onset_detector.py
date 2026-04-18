import numpy as np
import librosa
import scipy.ndimage
import matplotlib.pyplot as plt
import os

# Import our shared feature extraction pipeline
from features import compute_log_mel_spectrogram

class OnsetDetectorLFSF:
    def __init__(self, sr=44100):
        self.sr = sr
        # Slide 66: 10ms hop size, 23ms window size
        self.hop_length = int(self.sr * 0.010)  # 441 samples
        self.n_fft = 1024                       # ~23ms is 1014 samples, 1024 is standard power of 2
        self.n_mels = 80                        # Standard number of mel bands
        self.lambda_val = 100.0                 # Log compression factor

    def compute_detection_function(self, audio, return_spec=False):
        """
        Implements the LogFiltSpecFlux detection function from Lecture 4.
        If return_spec is True, it returns the 2D log-mel spectrogram for CNN training.
        Otherwise, it returns the 1D smoothed activation function.
        """
        # 1 & 2. REFACTORED: Compute Log-Compressed Mel Spectrogram using features.py
        log_mel_spec = compute_log_mel_spectrogram(
            audio=audio, 
            sr=self.sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels,
            lambda_val=self.lambda_val
        )
        
        # --- CNN EXIT POINT ---
        # If we are training the CNN, we stop here and return the 2D image
        if return_spec:
            return log_mel_spec
        
        # 3. First-Order Difference (Spectral Flux)
        # Pad with a zero column at the start so shifting doesn't lose the first frame
        log_mel_spec_padded = np.pad(log_mel_spec, ((0, 0), (1, 0)), mode='constant')
        diff = log_mel_spec - log_mel_spec_padded[:, :-1]
        
        # 4. Half-Wave Rectification: H(x) = max(x, 0)
        hwr_diff = np.maximum(diff, 0)
        
        # 5. Sum across frequency bins to get a 1D activation function
        activation = np.sum(hwr_diff, axis=0)
        
        # 6. Smooth the activation function to prevent double-triggering on jagged peaks
        activation = scipy.ndimage.gaussian_filter1d(activation, sigma=1.5)
        
        return activation

    def pick_peaks(self, activation, pre_max=3, post_max=3, pre_avg=15, post_avg=15, delta=1.0, wait=5):
        """
        Implements Adaptive Thresholding.
        Finds peaks that are local maxima AND above a moving average threshold.
        """
        # 1. Local Maximum condition
        local_max = scipy.ndimage.maximum_filter1d(activation, size=pre_max + post_max + 1)
        is_max = (activation == local_max)
        
        # 2. Adaptive Threshold condition (Mean + delta)
        local_mean = scipy.ndimage.uniform_filter1d(activation, size=pre_avg + post_avg + 1)
        is_above_threshold = (activation >= (local_mean + delta))
        
        # Combine conditions
        peak_mask = is_max & is_above_threshold
        peak_frames = np.where(peak_mask)[0]
        
        # 3. Minimum distance (wait) condition
        valid_peaks = []
        last_peak = -wait
        for p in peak_frames:
            if p - last_peak >= wait:
                valid_peaks.append(p)
                last_peak = p
                
        # Convert frame indices back to timestamps in seconds
        onset_times = librosa.frames_to_time(valid_peaks, sr=self.sr, hop_length=self.hop_length)
        return onset_times, activation, local_mean + delta

# ==========================================
# SANITY CHECK: Let's see your smoothed algorithm!
# ==========================================
if __name__ == "__main__":
    from data_loader import AMPDataLoader
    
    # Make sure this path matches where your data actually is!
    DATA_DIR = os.path.join("..", "data", "train", "train") 
    loader = AMPDataLoader(data_dir=DATA_DIR)
    
    try:
        track_data = loader.load_track("ff123_2nd_vent_clip")
        
        detector = OnsetDetectorLFSF(sr=track_data['sr'])
        
        print("Computing detection function...")
        # LFSF call (default behavior)
        activation = detector.compute_detection_function(track_data['audio'])
        
        print("Picking peaks...")
        # Using the best parameters found in your 0.77 F1 cross-validation
        predicted_onsets, raw_activation, threshold_curve = detector.pick_peaks(
            activation, 
            delta=1.0, 
            wait=5      
        )
        
        print(f"Algorithm predicted {len(predicted_onsets)} onsets.")
        print(f"Ground truth has {len(track_data['onsets'])} onsets.")
        
        # Create a time array and calculate magnitude once
        time_audio = np.linspace(0, track_data['duration_sec'], len(track_data['audio']))
        audio_magnitude = np.abs(track_data['audio'])
        
        # Create 2 plots stacked vertically
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)
        
        # --- Plot 1: Ground Truth Only ---
        ax1.plot(time_audio, audio_magnitude, label="Audio Magnitude", color='gray', alpha=0.6)
        ax1.vlines(track_data['onsets'], ymin=0, ymax=1, color='red', linestyle='--', alpha=0.8, label='Ground Truth')
        ax1.set_title("Actual Music: Ground Truth Onsets")
        ax1.legend(loc="upper right")
        
        # --- Plot 2: Predictions Only ---
        ax2.plot(time_audio, audio_magnitude, label="Audio Magnitude", color='gray', alpha=0.6)
        ax2.vlines(predicted_onsets, ymin=0, ymax=1, color='blue', alpha=0.7, label='Predicted')
        ax2.set_title("Your Algorithm: Predicted Onsets")
        ax2.set_xlabel("Time (seconds)")
        ax2.legend(loc="upper right")
        
        plt.xlim(0, 10) 
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Could not run test block. Error: {e}")