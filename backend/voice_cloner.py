import sys
import numpy as np
import librosa
from scipy.io.wavfile import write
from pathlib import Path

# Path to the Real-Time-Voice-Cloning repo
sys.path.append("C:/Users/Vivek Vasani/Desktop/voice-style-learner/Real-Time-Voice-Cloning")  # Adjust this to your actual path

from encoder import inference as encoder_infer
from synthesizer import inference as synthesizer_infer
from vocoder import inference as vocoder_infer  # Correct import for vocoder

def clone_voice_with_model(text, audio_files):
    """
    This function takes the text to be synthesized and a list of audio files for training.
    It will synthesize the voice from the given preprocessed files using the pre-trained voice model.
    """
    # Path to the pre-trained synthesizer model
    synthesizer_model_path = "C:/Users/Vivek Vasani/Desktop/voice-style-learner/Real-Time-Voice-Cloning/synthesizer/saved_models/default/synthesizer_model.pt"
    
    # Load pre-trained models from the repo
    synthesizer = synthesizer_infer.Synthesizer(synthesizer_model_path)

    # Check vocoder initialization, adjusting as per the actual class
    try:
        vocoder = vocoder_infer.WaveGlowVocoder()  # Replace with the actual class name
    except AttributeError:
        print("Vocoder initialization failed, check the available classes in 'vocoder/inference.py'")

    # Prepare audio for encoding and extract speaker embeddings
    speaker_embeddings = []
    for audio_file in audio_files:
        audio, _ = librosa.load(audio_file, sr=16000)
        # Use the pre-trained encoder to get speaker embeddings
        speaker_embedding = encoder_infer.embed_utterance(audio)  # Correct usage of encoder function
        speaker_embeddings.append(speaker_embedding)

    # Synthesize voice for the input text with the target speaker
    mel_output, alignment = synthesizer.encode(text, speaker_embeddings[0])  # Correct method
    generated_audio = vocoder.decode(mel_output)  # Use vocoder to decode mel spectrogram to audio

    # Save the generated audio
    output_file = "cloned_voice_output.wav"
    write(output_file, 16000, np.array(generated_audio))
    print(f"Cloned voice saved as: {output_file}")

# Example usage
audio_files = [str(file) for file in Path('C:/Users/Vivek Vasani/Desktop/voice-style-learner/data/preprocessed_audio').glob('*.wav')]
text_to_synthesize = "Hello, this is your cloned voice speaking."

clone_voice_with_model(text_to_synthesize, audio_files)
