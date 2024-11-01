import gradio as gr
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torchaudio

# Define emotion labels and corresponding icons
emotion_labels = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
emotion_icons = {
    "angry": "😠", "calm": "😌", "disgust": "🤢", "fearful": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprised": "😲"
}

# Load model and processor
model_name = "Dpngtm/wav2vec2-emotion-recognition"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name, num_labels=len(emotion_labels))

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def recognize_emotion(audio):
    try:
        # Handle case where no audio is provided
        if audio is None:
            return {f"{emotion} {emotion_icons[emotion]}": 0.0 for emotion in emotion_labels}
        
        # Load and preprocess the audio
        audio_path = audio if isinstance(audio, str) else audio.name
        speech_array, sampling_rate = torchaudio.load(audio_path)
        
        # Limit audio length to 1 minute (60 seconds)
        duration = speech_array.shape[1] / sampling_rate
        if duration > 60:
            return {
                "Error": "Audio too long (max 1 minute)",
                **{f"{emotion} {emotion_icons[emotion]}": 0.0 for emotion in emotion_labels}
            }
        
        # Resample audio if not at 16kHz
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            speech_array = resampler(speech_array)
        
        # Convert stereo to mono if necessary
        if speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)
        
        # Normalize audio
        speech_array = speech_array / torch.max(torch.abs(speech_array))
        speech_array = speech_array.squeeze().numpy()
        
        # Process audio with the model
        inputs = processor(speech_array, sampling_rate=16000, return_tensors='pt', padding=True)
        input_values = inputs.input_values.to(device)
        
        with torch.no_grad():
            outputs = model(input_values)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            
            # Prepare the confidence scores without converting to percentages
            confidence_scores = {
                f"{emotion} {emotion_icons[emotion]}": prob
                for emotion, prob in zip(emotion_labels, probs)
            }
            
            # Sort scores in descending order
            sorted_scores = dict(sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True))
            return sorted_scores
    
    except Exception as e:
        # Return error message along with zeroed-out emotion scores
        return {
            "Error": str(e),
            **{f"{emotion} {emotion_icons[emotion]}": 0.0 for emotion in emotion_labels}
        }

# Supported emotions for display
supported_emotions = " | ".join([f"{emotion_icons[emotion]} {emotion}" for emotion in emotion_labels])

# Gradio Interface setup
interface = gr.Interface(
    fn=recognize_emotion,
    inputs=gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        label="Record or Upload Audio"
    ),
    outputs=gr.Label(
        num_top_classes=len(emotion_labels),
        label="Detected Emotion"
    ),
    title="Speech Emotion Recognition",
    description=f"""
    ### Supported Emotions:
    {supported_emotions}
    
    Maximum audio length: 1 minute""",
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="blue"
    ),
    css="""
        .gradio-container {max-width: 800px}
        .label {font-size: 18px}
    """
)

if __name__ == "__main__":
    interface.launch(
        share=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860
    )
