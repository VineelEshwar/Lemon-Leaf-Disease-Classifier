# app/app.py

import streamlit as st
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO

from model import EfficientNetModel
from utils import preprocess_image

# Lemon disease labels
lemon_labels = [
    "Anthracnose", "Bacterial Blight", "Citrus Canker", "Curl Virus",
    "Deficiency Leaf", "Dry Leaf", "Healthy Leaf", "Sooty Mould",
    "Spider Mites"
]

# Fertilizer + disease guide
fertilizer_guide = {
    "Anthracnose": {
        "desc": "Fungal infection causing dark, sunken lesions.",
        "fertilizer": "Use copper-based fungicides. Prune and improve air circulation."
    },
    "Bacterial Blight": {
        "desc": "Water-soaked spots and leaf drop.",
        "fertilizer": "Apply streptomycin-based sprays. Ensure sanitation."
    },
    "Citrus Canker": {
        "desc": "Scabby lesions and leaf yellowing.",
        "fertilizer": "Copper sprays. Remove infected material."
    },
    "Curl Virus": {
        "desc": "Leaf curling caused by viral infection.",
        "fertilizer": "Control whiteflies. Use resistant stock."
    },
    "Deficiency Leaf": {
        "desc": "Yellowing or spotting due to nutrient lack.",
        "fertilizer": "Add balanced NPK, especially nitrogen, magnesium."
    },
    "Dry Leaf": {
        "desc": "Crisp edges, drought-like symptoms.",
        "fertilizer": "Improve irrigation. Use Zn & Fe supplements."
    },
    "Healthy Leaf": {
        "desc": "No disease detected.",
        "fertilizer": "No treatment needed. Continue normal care."
    },
    "Sooty Mould": {
        "desc": "Black fungal coating from insect honeydew.",
        "fertilizer": "Clean leaves. Control aphids, whiteflies, scale."
    },
    "Spider Mites": {
        "desc": "Tiny webs and leaf speckling.",
        "fertilizer": "Use miticides. Maintain humidity."
    }
}

# Load model
@st.cache_resource
def load_model():
    model = EfficientNetModel(num_classes=9)
    model_path = os.path.join("..", "Model", "effnet_for_lemon_50.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Page settings
st.set_page_config(page_title="🍋 Lemon Leaf Disease Detector", layout="centered")

# App Header
st.markdown("<h1 style='text-align: center;'>🍋 Lemon Leaf Disease Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a leaf image to identify diseases and receive treatment recommendations.</p>", unsafe_allow_html=True)

# Sidebar info
st.sidebar.header("📌 Instructions")
st.sidebar.markdown("""
- Upload a **clear lemon leaf image**.
- Wait a few seconds for prediction.
- Scroll to see disease explanation & fertilizer advice.
""")

# File uploader
uploaded_file = st.file_uploader("📷 Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Show image in smaller layout
    st.image(image, caption="Uploaded Leaf", use_container_width=False, width=300)

    with st.spinner("🧠 Predicting..."):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = torch.softmax(outputs, dim=1)[0][predicted_class].item()
            all_probs = torch.softmax(outputs, dim=1)[0].numpy()

    predicted_label = lemon_labels[predicted_class]
    st.success(f"### ✅ Prediction: **{predicted_label}**")
    st.metric("Confidence Score", f"{confidence*100:.2f}%")

    # Show confidence in table
    st.subheader("📊 Class Probabilities")
    prob_df = pd.DataFrame({
        "Disease": lemon_labels,
        "Confidence": [f"{p*100:.2f}%" for p in all_probs]
    }).sort_values(by="Confidence", ascending=False).reset_index(drop=True)
    st.table(prob_df)

    # Fertilizer recommendation
    st.subheader("🌿 Diagnosis & Fertilizer Advice")
    info = fertilizer_guide.get(predicted_label, None)
    if info:
        st.markdown(f"**🧬 Description:** {info['desc']}")
        st.info(f"💊 **Fertilizer / Treatment Tip:** {info['fertilizer']}")
    else:
        st.warning("No treatment information available.")

    # Prediction history
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "class": predicted_label,
        "confidence": f"{confidence*100:.2f}%"
    })

    with st.expander("🕘 View Past Predictions"):
        for item in st.session_state.history[-5:][::-1]:
            st.write(f"→ **{item['class']}** ({item['confidence']})")

    # Export as PDF or markdown
    def generate_download_link(text, filename):
        b64 = base64.b64encode(text.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Download Report</a>'
        return href

    report = f"""
Lemon Leaf Disease Report
===========================
Prediction: {predicted_label}
Confidence: {confidence*100:.2f}%

Description:
{info['desc'] if info else 'N/A'}

Fertilizer Advice:
{info['fertilizer'] if info else 'N/A'}
"""
    st.markdown(generate_download_link(report, "lemon_disease_report.txt"), unsafe_allow_html=True)

# Expandable info section
with st.expander("ℹ️ About this App"):
    st.markdown("""
- 📄 **Paper Published (IEEE Conference)**:https://ieeexplore.ieee.org/document/11064055                 
- 🧠 **Model**: EfficientNet B0 with Custom Layers made (PyTorch, custom classifier)
- 🧪 **Classes**: 9 lemon leaf conditions (healthy + 8 diseases)
- 📂 **Model Size**: ~15MB
- 📸 **Input**: JPG/PNG image of a lemon leaf
- 🛠️ **Built With**: Streamlit, PyTorch, Matplotlib
- 📄 **License**: MIT License
- 📧 **Contact**: [vineelesh123@gmail.com]
""")
