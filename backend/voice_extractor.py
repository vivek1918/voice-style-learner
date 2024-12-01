import os
import numpy as np
import soundfile as sf
import librosa
import uuid
from pathlib import Path
import noisereduce as nr
from moviepy.editor import VideoFileClip  # Import moviepy for extracting audio

class VoiceExtractor:
    def __init__(self, base_dir='C:/Users/Vivek Vasani/Desktop/voice-style-learner/data'):
        """
        Initialize VoiceExtractor with configurable base directory

        Args:
            base_dir (str): Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.videos_dir = self.base_dir / 'videos'
        self.audio_dir = self.base_dir / 'processed_audio'

        # Create necessary directories
        for dir_path in [self.audio_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def process_all_videos(self, sample_rate=16000):
        """
        Process all videos in the videos directory to extract clean voice segments

        Args:
            sample_rate (int): Target audio sample rate

        Returns:
            list: List of processed voice information for each video
        """
        processed_results = []

        for video_file in self.videos_dir.glob("*.mp4"):
            print(f"Processing video: {video_file.name}")
            result = self.process_video_for_voice_learning(video_file.name, sample_rate)
            if result:
                processed_results.append(result)

        return processed_results

    def process_video_for_voice_learning(self, video_filename, sample_rate=16000):
        """
        Process video to extract clean voice segments

        Args:
            video_filename (str): Name of the video file in the videos directory
            sample_rate (int): Target audio sample rate

        Returns:
            dict: Processed voice information
        """
        video_path = self.videos_dir / Path(video_filename)

        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            return None

        try:
            # Extract audio from video using moviepy
            video_clip = VideoFileClip(str(video_path))
            audio_clip = video_clip.audio
            audio_path = str(self.audio_dir / f"{uuid.uuid4()}.wav")
            audio_clip.write_audiofile(audio_path)

            # Now, load the audio file with librosa
            audio, orig_sr = librosa.load(audio_path, sr=sample_rate)

            if len(audio) == 0:
                print(f"No audio data found in {video_path}")
                return None

            # Noise reduction
            reduced_noise = nr.reduce_noise(
                y=audio, sr=sample_rate, prop_decrease=0.7
            )

            # Normalize audio
            normalized_audio = librosa.util.normalize(reduced_noise)

            # Generate unique filename for processed audio
            audio_filename = f"voice_{uuid.uuid4()}.wav"
            processed_audio_path = self.audio_dir / audio_filename

            # Save processed audio
            sf.write(str(processed_audio_path), normalized_audio, sample_rate)

            # Detect voice segments
            segments = self._detect_voice_segments(normalized_audio, sample_rate)

            return {
                'full_audio_path': processed_audio_path,
                'voice_segments': segments,
                'sample_rate': sample_rate
            }

        except Exception as e:
            print(f"Video processing error for {video_path}: {e}")
            return None

    def _detect_voice_segments(self, audio, sample_rate=16000, min_segment_length=1.0):
        """
        Advanced voice activity detection

        Args:
            audio (np.ndarray): Input audio signal
            sample_rate (int): Audio sample rate
            min_segment_length (float): Minimum segment length

        Returns:
            list: Voice segment timestamps
        """
        # Compute various features for robust detection
        rms = librosa.feature.rms(y=audio)[0]
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]

        # Adaptive thresholding
        rms_threshold = np.percentile(rms, 75)
        zcr_threshold = np.percentile(zcr, 70)

        segments = []
        in_speech = False
        start_time = 0

        for i in range(len(rms)):
            time = i / sample_rate

            # Complex voice activity condition
            is_voice = (rms[i] > rms_threshold) and (zcr[i] < zcr_threshold)

            if is_voice and not in_speech:
                start_time = time
                in_speech = True

            if not is_voice and in_speech:
                segment_duration = time - start_time
                if segment_duration >= min_segment_length:
                    segments.append((start_time, time))
                in_speech = False

        return segments

# Example usage
if __name__ == "__main__":
    extractor = VoiceExtractor()

    # Process all videos from the pre-downloaded folder
    processed_results = extractor.process_all_videos()

    if processed_results:
        for result in processed_results:
            print("Processed data:", result)
