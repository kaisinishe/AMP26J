import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt

class AMPDataLoader:
    def __init__(self, data_dir, sample_rate=44100):
        """
        Initializes the data loader and indexes all available tracks.
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        
        # Scan directory for all .wav files and extract their base names
        wav_files = glob.glob(os.path.join(self.data_dir, '*.wav'))
        self.track_ids = sorted([os.path.splitext(os.path.basename(f))[0] for f in wav_files])
        
        print(f"Initialized DataLoader: Found {len(self.track_ids)} tracks in {data_dir}")

    def _read_timestamp_file(self, filepath):
        """Reads a .gt file and extracts the first column as timestamps (handles onsets and beats)."""
        if not os.path.exists(filepath):
            return np.array([])
        
        timestamps = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    # Always grab just the first column (the timestamp in seconds)
                    try:
                        timestamps.append(float(parts[0]))
                    except ValueError:
                        pass # Skip safely if there's a malformed line
                    
        return np.array(timestamps)

    def _read_tempo_file(self, filepath):
        """Reads the tempo.gt file which can have 1 to 3 values on a single line."""
        if not os.path.exists(filepath):
            return []
        with open(filepath, 'r') as f:
            line = f.readline().strip()
            if line:
                return [float(x) for x in line.split() if x]
            return []

    def load_track(self, track_id):
        """
        Loads the audio array and all corresponding ground truth data for a specific track.
        """
        if track_id not in self.track_ids:
            raise ValueError(f"Track '{track_id}' not found in dataset.")

        wav_path = os.path.join(self.data_dir, f"{track_id}.wav")
        onset_path = os.path.join(self.data_dir, f"{track_id}.onsets.gt")
        beat_path = os.path.join(self.data_dir, f"{track_id}.beats.gt")
        tempo_path = os.path.join(self.data_dir, f"{track_id}.tempo.gt")

        # Load Audio (forces mono and standardizes sample rate)
        audio, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True)

        return {
            'track_id': track_id,
            'audio': audio,
            'sr': sr,
            'duration_sec': len(audio) / sr,
            'onsets': self._read_timestamp_file(onset_path),
            'beats': self._read_timestamp_file(beat_path),
            'tempo': self._read_tempo_file(tempo_path)
        }

# ==========================================
# SANITY CHECK & VERIFICATION
# ==========================================
if __name__ == "__main__":
    # Ensure this points to where you extracted the train folder
    # Adjust this path based on your actual directory structure!
    DATA_DIR = os.path.join("..", "data", "train", "train") 
    
    loader = AMPDataLoader(data_dir=DATA_DIR)
    
    if len(loader.track_ids) > 0:
        # 1. Load the specific track from your screenshot
        test_track = "ff123_2nd_vent_clip"
        
        if test_track in loader.track_ids:
            print(f"\nLoading track: {test_track}...")
            
            try:
                track_data = loader.load_track(test_track)
                
                # 2. Print statistics to check if arrays match your expectations
                print(f"Audio Duration: {track_data['duration_sec']:.2f} seconds")
                print(f"Sample Rate: {track_data['sr']} Hz")
                print(f"Total Onsets Found: {len(track_data['onsets'])}")
                print(f"First 5 Onsets: {track_data['onsets'][:5]}")
                print(f"Total Beats Found: {len(track_data['beats'])}")
                print(f"First 5 Beats: {track_data['beats'][:5]}")
                print(f"Tempo Annotations: {track_data['tempo']}")
                
                # 3. Visual Verification
                print("\nGenerating plot for visual verification. Close the plot window to exit.")
                
                # Create a time array for the x-axis
                time_axis = np.linspace(0, track_data['duration_sec'], len(track_data['audio']))
                
                plt.figure(figsize=(12, 4))
                plt.plot(time_axis, track_data['audio'], label='Audio Waveform', alpha=0.6, color='blue')
                
                # Plot ground truth onsets as red vertical lines
                for onset in track_data['onsets']:
                    plt.axvline(x=onset, color='red', linestyle='--', alpha=0.8, linewidth=1)
                    
                plt.title(f"Waveform and True Onsets for {test_track}")
                plt.xlabel("Time (seconds)")
                plt.ylabel("Amplitude")
                plt.xlim(0, 10) # Zoom in on the first 10 seconds to see clearly
                plt.legend(['Waveform', 'True Onsets'])
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Error loading track: {e}")
        else:
            print(f"Track '{test_track}' not found in {DATA_DIR}. Please check the path.")
    else:
        print("No wav files found. Check your DATA_DIR path!")