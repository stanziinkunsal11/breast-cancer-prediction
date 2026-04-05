import numpy as np
import gradio as gr
import joblib

# Load sklearn model
model = joblib.load("breastCancerModel.pkl")

def predict_cancer(*inputs):
    # Convert inputs to numpy array
    input_data = np.array(inputs).astype(float).reshape(1, -1)

    # Prediction
    prediction = model.predict(input_data)[0]

    if prediction == 0:
        return "Malignant (Cancerous)"
    else:
        return "Benign (Non-Cancerous)"

feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]
# Create 30 input fields
inputs = [gr.Number(label=name) for name in feature_names]

# Gradio interface
interface = gr.Interface(
    fn=predict_cancer,
    inputs=inputs,
    outputs="text",
    title="Breast Cancer Prediction",
    description="Enter 30 medical features to classify tumor as malignant or benign"
)

interface.launch()