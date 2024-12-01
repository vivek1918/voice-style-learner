import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os
import glob

class VoiceStyleDataset(Dataset):
    def __init__(self, audio_files):
        """
        Dataset for voice style learning
        
        Args:
            audio_files (list): List of audio file paths
        """
        self.audio_files = audio_files
        self.audio_features = self._extract_features()

    def _extract_features(self):
        """
        Extract mel spectrograms and other features
        
        Returns:
            list: List of extracted audio features
        """
        features = []
        for audio_path in self.audio_files:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_mels=128, 
                fmax=8000
            )
            
            # Convert to log mel spectrogram
            log_mel_spec = librosa.power_to_db(mel_spec)
            features.append(log_mel_spec)
        
        return features

    def __len__(self):
        return len(self.audio_features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.audio_features[idx])

class VoiceStyleTransformer(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        """
        Neural network for voice style transformation
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
        """
        super().__init__()
        
        self.transformer = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, style_embedding):
        """
        Forward pass with style embedding
        
        Args:
            x (torch.Tensor): Input features
            style_embedding (torch.Tensor): Voice style embedding
        
        Returns:
            torch.Tensor: Transformed features
        """
        # Concatenate style embedding with input
        style_expanded = style_embedding.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x_with_style = torch.cat([x, style_expanded], dim=1)
        
        return self.transformer(x_with_style)

class VoiceStyleLearner:
    def __init__(self, learning_rate=1e-4):
        """
        Initialize voice style learning system
        
        Args:
            learning_rate (float): Optimizer learning rate
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = VoiceStyleTransformer().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train(self, audio_files, epochs=50):
        """
        Train voice style transformation model
        
        Args:
            audio_files (list): List of audio file paths
            epochs (int): Number of training epochs
        
        Returns:
            dict: Training metrics
        """
        dataset = VoiceStyleDataset(audio_files)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        training_history = {
            'epoch_losses': [],
            'final_loss': None
        }
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Random style embedding
                style_embedding = torch.randn(batch.size(0), 128).to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                transformed_batch = self.model(batch, style_embedding)
                
                # Compute loss
                loss = self.criterion(transformed_batch, batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Store epoch loss
            epoch_loss = total_loss / len(dataloader)
            training_history['epoch_losses'].append(epoch_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")
        
        # Final loss
        training_history['final_loss'] = training_history['epoch_losses'][-1]
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': training_history
        }, 'models/voice_style_model.pth')
        
        return training_history

    @classmethod
    def load_model(cls, model_path='models/voice_style_model.pth'):
        """
        Load a pre-trained voice style model
        
        Args:
            model_path (str): Path to saved model
        
        Returns:
            VoiceStyleLearner: Instantiated model
        """
        # Create an instance
        learner = cls()
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=learner.device)
        
        # Load model and optimizer states
        learner.model.load_state_dict(checkpoint['model_state_dict'])
        learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return learner

    @staticmethod
    def find_audio_files(directory, extensions=['.wav', '.mp3', '.ogg']):
        """
        Find all audio files in a given directory
        
        Args:
            directory (str): Directory to search
            extensions (list): File extensions to include
        
        Returns:
            list: List of audio file paths
        """
        audio_files = []
        for ext in extensions:
            audio_files.extend(glob.glob(os.path.join(directory, f'**/*{ext}'), recursive=True))
        
        return audio_files

# Example usage
def main():
    # Create an instance of VoiceStyleLearner
    learner = VoiceStyleLearner()
    
    # Find audio files in a specific directory
    audio_files = VoiceStyleLearner.find_audio_files("C:/Users/Vivek Vasani/Desktop/voice-style-learner/data/preprocessed_audio")
    
    # Check if we have enough audio files
    if len(audio_files) > 0:
        # Train the model
        training_results = learner.train(audio_files, epochs=10)
        print("Training complete. Final Loss:", training_results['final_loss'])
        
        # Optional: Load the trained model
        loaded_learner = VoiceStyleLearner.load_model()
        print("Model loaded successfully!")
    else:
        print("No audio files found for training.")

if __name__ == "__main__":
    main()