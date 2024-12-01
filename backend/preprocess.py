import os
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from pathlib import Path

class AudioPreprocessor:
    def __init__(self, audio_dir='C:/Users/Vivek Vasani/Desktop/voice-style-learner/data/processed_audio', 
                 output_dir='C:/Users/Vivek Vasani/Desktop/voice-style-learner/data/preprocessed_audio', sample_rate=16000):
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_audio_files(self):
        """
        Preprocess all audio files in the processed_audio directory by applying noise reduction, resampling, and normalization.

        Returns:
            List of paths to the processed audio files.
        """
        processed_files = []

        for audio_file in self.audio_dir.glob("*.wav"):
            print(f"Processing audio: {audio_file.name}")
            result = self.process_audio(audio_file)
            if result:
                processed_files.append(result)

        return processed_files

    def process_audio(self, audio_path):
        """
        Process a single audio file with noise reduction, resampling, and normalization.

        Args:
            audio_path (Path): Path to the audio file.

        Returns:
            str: Path to the processed audio file.
        """
        try:
            # Load audio file
            audio, sr = librosa.load(str(audio_path), sr=None)  # Load with original sample rate

            # Debug: Check if audio loaded properly
            print(f"Audio loaded: {audio.shape} samples, Sample rate: {sr}")

            # Noise reduction
            audio_reduced = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.7)

            # Debug: Check if noise reduction has been applied
            print(f"Audio after noise reduction: {audio_reduced.shape} samples")

            # Resample to desired sample rate
            if sr != self.sample_rate:
                audio_reduced = librosa.resample(audio_reduced, orig_sr=sr, target_sr=self.sample_rate)
                print(f"Resampled audio to: {self.sample_rate} Hz")

            # Normalize audio
            audio_normalized = librosa.util.normalize(audio_reduced)
            print(f"Audio normalized. Max amplitude: {np.max(audio_normalized)}")

            # Save processed audio to the output directory
            processed_audio_path = self.output_dir / f"processed_{audio_path.name}"
            sf.write(str(processed_audio_path), audio_normalized, self.sample_rate)

            print(f"Processed audio saved: {processed_audio_path}")
            return processed_audio_path

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None


# Example usage
if __name__ == "__main__":
    preprocessor = AudioPreprocessor()

    # Preprocess all audio files
    processed_audio_files = preprocessor.preprocess_audio_files()

    if processed_audio_files:
        print("Processed audio files:")
        for processed_file in processed_audio_files:
            print(processed_file)
