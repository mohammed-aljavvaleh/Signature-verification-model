import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
import sys
sys.path.insert(0, '.')

from src.config import Config
from src.model import SiameseNetwork

st.set_page_config(page_title="Signature Verification", page_icon="✍️")

@st.cache_resource
def load_model():
    model = SiameseNetwork(embedding_dim=Config.EMBEDDING_DIM)
    model_path = Config.MODELS_DIR / Config.BEST_MODEL_NAME
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()
    return model, checkpoint

def preprocess_image(image):
    img = np.array(image)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (220, 155))
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return img_tensor

def verify_signatures(model, img1, img2, threshold=0.1302):
    img1_tensor = preprocess_image(img1).to(Config.DEVICE)
    img2_tensor = preprocess_image(img2).to(Config.DEVICE)
    
    with torch.no_grad():
        embedding1, embedding2 = model(img1_tensor, img2_tensor)
        distance = F.pairwise_distance(embedding1, embedding2).item()
    
    is_genuine = distance < threshold
    if is_genuine:
        confidence = min(100, max(0, (1 - distance / threshold) * 100))
    else:
        confidence = min(100, max(0, ((distance - threshold) / threshold) * 100))
    
    return is_genuine, distance, confidence

# Main app
st.title("✍️ Signature Verification AI")
st.write("Upload two signatures to verify if they match")

# Load model
with st.spinner("Loading model..."):
    model, checkpoint = load_model()

st.success("✅ Model loaded successfully!")

# Sidebar
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Threshold", 0.0, 0.3, 0.1302, 0.01)
    st.info(f"Model trained for {checkpoint['epoch']+1} epochs")

# Upload images
col1, col2 = st.columns(2)

with col1:
    st.subheader("Reference Signature")
    img1_file = st.file_uploader("Upload reference", type=['png', 'jpg', 'jpeg'], key="img1")
    if img1_file:
        img1 = Image.open(img1_file)
        st.image(img1, use_container_width=True)

with col2:
    st.subheader("Signature to Verify")
    img2_file = st.file_uploader("Upload signature", type=['png', 'jpg', 'jpeg'], key="img2")
    if img2_file:
        img2 = Image.open(img2_file)
        st.image(img2, use_container_width=True)

# Verify button
if img1_file and img2_file:
    if st.button("🔍 Verify Signatures", type="primary"):
        with st.spinner("Analyzing..."):
            is_genuine, distance, confidence = verify_signatures(model, img1, img2, threshold)
            
            st.markdown("---")
            
            if is_genuine:
                st.success("✅ GENUINE SIGNATURE")
            else:
                st.error("❌ FORGED SIGNATURE")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Distance", f"{distance:.4f}")
            col_m2.metric("Threshold", f"{threshold:.4f}")
            col_m3.metric("Confidence", f"{confidence:.1f}%")
