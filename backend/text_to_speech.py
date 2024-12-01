import torch
import torchaudio
from TTS.api import TTS
import os

class TextToSpeechGenerator:
    def __init__(self, model_path='models/tts_model.pth'):
        """
        Initialize text-to-speech generator
        
        Args:
            model_path (str): Path to pre-trained TTS model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use Coqui TTS for high-quality text-to-speech
        self.tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", 
                             progress_bar=False).to(self.device)
        
        # Directory for generated speech
        os.makedirs('data/generated_speech', exist_ok=True)

    def generate_speech(self, text, speaker_wav=None, language='en'):
        """
        Generate speech from text
        
        Args:
            text (str): Text to convert to speech
            speaker_wav (str, optional): Reference audio for voice cloning
            language (str, default='en'): Language of the text
        
        Returns:
            str: Path to generated audio file
        """
        try:
            # Generate unique filename
            output_path = f"data/generated_speech/speech_{hash(text)}.wav"
            
            # Generate speech
            self.tts_model.tts_to_file(
                text=text, 
                file_path=output_path, 
                speaker_wav=speaker_wav,
                language=language
            )
            
            return output_path
        
        except Exception as e:
            print(f"Speech generation error: {e}")
            return None

# Example usage
if __name__ == "__main__":
    tts = TextToSpeechGenerator()
    speech_file = tts.generate_speech("Hello, this is a voice cloning demonstration!")
    print(f"Generated speech at: {speech_file}")