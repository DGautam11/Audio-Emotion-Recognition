import torch
import librosa
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForAudioClassification, AutoProcessor

class Inference:
    def __init__(self, model_name: str = "Dpngtm/wav2vec2-emotion-recognition", device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        
        self.emotion_icons = {
            "angry": "😠", 
            "disgust": "🤢", 
            "fear": "😨", "fearful": "😨",
            "happy": "😊", 
            "neutral": "😐",
            "sad": "😢", 
            "surprise": "😲", "surprised": "😲"
        }

    def predict(self, audio_path: str):
        if not audio_path:
            return None

        try:
           
            speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

            speech_array = torch.tensor(speech_array)

            # Handle stereo (if librosa returns 2 channels)
            if len(speech_array.shape) > 1:
                 # Librosa loads as (channels, time) if mono=False, but default is mono=True
                 # If it somehow loaded stereo, average it.
                 speech_array = torch.mean(speech_array, dim=0)

            inputs = self.processor(
                speech_array.numpy(), 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

            results = {}
            for i, score in enumerate(probs):
                label = self.id2label[i]
                icon = self.emotion_icons.get(label.lower(), "")
                results[f"{label} {icon}"] = float(score)

            return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

        except Exception as e:
            
            print(f"Prediction Error: {e}")
            return {"Error": 0.0} 