import librosa
import numpy as np

def compute_log_mel_spectrogram(audio, sr, n_fft=1024, hop_length=441, n_mels=80, lambda_val=100.0):
    """
    Computes a Log-Compressed Mel Spectrogram.
    This is standard preprocessing for Onsets, Beats, and Tempo.
    """
    # 1. Compute Mel Spectrogram (Magnitude)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels,
        fmin=27.5,      # A0
        fmax=16000.0    # Typical upper bound
    )
    
    # 2. Logarithmic Compression
    log_mel_spec = np.log(1 + lambda_val * mel_spec)
    
    return log_mel_spec