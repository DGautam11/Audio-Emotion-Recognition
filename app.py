import gradio as gr
from src.inference import Inference

inf = Inference()

def process_audio(audio):
    return inf.predict(audio)

supported_emotions = " | ".join(sorted(set([f"{k} {v}" for k, v in inf.emotion_icons.items()])))

interface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        label="Record or Upload Audio"
    ),
    outputs=gr.Label(
        num_top_classes=3,
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
        server_name="0.0.0.0",
        server_port=7860
    )