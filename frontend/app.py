import streamlit as st
import os
import sys
sys.path.append('..')
from backend.voice_extractor import VoiceExtractor
from backend.voice_cloner import VoiceCloner
from backend.text_to_speech import TextToSpeechGenerator
from backend.style_learner import VoiceStyleLearner

class VoiceStyleApp:
    def __init__(self):
        # Initialize components
        self.extractor = VoiceExtractor()
        self.cloner = VoiceCloner()
        self.tts = TextToSpeechGenerator()
        self.style_learner = VoiceStyleLearner()

        # Ensure necessary directories exist
        os.makedirs('data/reference_audio', exist_ok=True)
        os.makedirs('data/generated_speech', exist_ok=True)
        os.makedirs('data/processed_audio', exist_ok=True)

    def download_youtube_video(self):
        st.header("🎥 YouTube Video Voice Extractor")
        
        # YouTube URL input
        youtube_url = st.text_input("Enter YouTube Video URL")
        
        if st.button("Download Video"):
            with st.spinner("Downloading video..."):
                try:
                    video_path = self.extractor.download_youtube_video(youtube_url)
                    
                    if video_path:
                        st.success(f"Video downloaded successfully: {video_path}")
                        
                        # Process video for voice
                        with st.spinner("Processing video for voice extraction..."):
                            processed_data = self.extractor.process_video_for_voice_learning(video_path)
                        
                        if processed_data:
                            st.success(f"Audio extracted: {processed_data['full_audio_path']}")
                            st.write("Voice Segments:", processed_data['voice_segments'])
                        else:
                            st.error("Failed to process video")
                    else:
                        st.error("Failed to download video")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    def voice_cloning(self):
        st.header("🎙️ Voice Cloning")
        
        # Reference audio upload
        reference_audio = st.file_uploader("Upload Reference Audio", type=['wav', 'mp3'])
        
        # Text input for speech generation
        text_to_speak = st.text_area("Enter text to generate speech")
        
        if st.button("Clone Voice"):
            if reference_audio and text_to_speak:
                with st.spinner("Cloning voice..."):
                    # Save reference audio
                    ref_audio_path = f"data/reference_audio/{reference_audio.name}"
                    with open(ref_audio_path, "wb") as f:
                        f.write(reference_audio.getbuffer())
                    
                    # Generate speech
                    speech_file = self.tts.generate_speech(text_to_speak, speaker_wav=ref_audio_path)
                    
                    if speech_file:
                        st.success("Voice cloned successfully!")
                        st.audio(speech_file)
                    else:
                        st.error("Failed to clone voice")
            else:
                st.warning("Please upload a reference audio and enter text")

    def train_voice_model(self):
        st.header("🧠 Train Voice Style Model")
        
        # Find audio files
        audio_files = VoiceStyleLearner.find_audio_files('data/processed_audio')
        
        st.write(f"Found {len(audio_files)} audio files for training")
        
        if st.button("Start Training"):
            if len(audio_files) > 0:
                with st.spinner("Training voice style model..."):
                    training_results = self.style_learner.train(audio_files)
                
                st.success("Training Complete!")
                st.write("Final Loss:", training_results['final_loss'])
                st.write("Training History:", training_results['epoch_losses'])
            else:
                st.error("No audio files found in data/processed_audio")

    def main_app(self):
        st.title("🎤 Voice Style Learner")
        
        # Sidebar navigation
        app_mode = st.sidebar.selectbox(
            "Choose Application Mode",
            ["YouTube Voice Extractor", "Voice Cloning", "Train Voice Model"]
        )

        if app_mode == "YouTube Voice Extractor":
            self.download_youtube_video()
        elif app_mode == "Voice Cloning":
            self.voice_cloning()
        elif app_mode == "Train Voice Model":
            self.train_voice_model()

def main():
    app = VoiceStyleApp()
    app.main_app()

if __name__ == "__main__":
    main()