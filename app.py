
!pip install gradio librosa --quiet

import gradio as gr
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class DummyScaler:
    def transform(self, X):
        return X
scaler = DummyScaler()

class DummyClassifier:
    def predict(self, X):
        return ['happy']
classifier = DummyClassifier()



emotion_images = {
    'happy': '/content/happy.jpeg',
    'sad': '/content/sad.jpeg',
    'angry': '/content/angry.jpeg'
}

def predict_emotion(audio):
    print("Received audio file path:", audio)
    signal, sr = librosa.load(audio, sr=None)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
    features = np.mean(mfccs, axis=1).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = classifier.predict(features_scaled)[0]
    print("Predicted emotion:", prediction)
    img_path = emotion_images.get(prediction, None)
    print("Image path:", img_path)
    return prediction, img_path

interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="filepath"),
    outputs=[
        gr.Textbox(label="Predicted Emotion"),
        gr.Image(type="filepath", label="Emotion Image")
    ],
    title="Speech Emotion Recognition",
    description="Upload an audio file to detect emotion (happy, sad, angry) and see the corresponding image."
)

interface.launch()
