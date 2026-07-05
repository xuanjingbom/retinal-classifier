import sys
import types
import os
import io
import base64
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as T
import anthropic

# Register custom ChannelAttention so best.pt loads correctly
class ChannelAttention(nn.Module):
    def __init__(self, channels=256, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels), nn.Sigmoid())
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

mod = types.ModuleType('ultralytics.nn.modules.channel_attention')
mod.ChannelAttention = ChannelAttention
sys.modules['ultralytics.nn.modules.channel_attention'] = mod
import ultralytics.nn.modules as _ult_mod
_ult_mod.ChannelAttention = ChannelAttention

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Retinal Disease Classifier", layout="wide")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ── Grad-CAM helper ───────────────────────────────────────────────────────────
class _Wrapper(nn.Module):
    def __init__(self, inner): super().__init__(); self.inner = inner
    def forward(self, x): return self.inner(x)

def compute_gradcam(pil_image, class_idx):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    inner = model.model
    for param in inner.parameters():
        param.requires_grad_(True)

    # find ChannelAttention layer; fall back to second-to-last layer
    target_layer = None
    for m in inner.model:
        if isinstance(m, ChannelAttention):
            target_layer = m
    if target_layer is None:
        target_layer = inner.model[-2]

    wrapper = _Wrapper(inner)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(pil_image).unsqueeze(0)

    with GradCAM(model=wrapper, target_layers=[target_layer]) as cam:
        mask = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(class_idx)])[0]

    rgb = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255.0
    from pytorch_grad_cam.utils.image import show_cam_on_image
    overlay = show_cam_on_image(rgb, mask, use_rgb=True)
    return Image.fromarray(overlay)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Retinal Disease Classification")
st.write("Upload a colour fundus photograph to classify the retinal condition.")

uploaded = st.file_uploader("Choose a fundus image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("Analysing..."):
        results = model(image, imgsz=224)
        probs   = results[0].probs
        names   = results[0].names
        top1    = probs.top1
        top1_conf = probs.top1conf.item()

    # ── Top row: image | prediction ──────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Fundus Image", use_container_width=True)
    with col2:
        st.markdown("### Prediction")
        st.success(f"**{names[top1]}**")
        st.metric("Confidence", f"{top1_conf:.1%}")

        st.markdown("#### All Class Probabilities")
        prob_list  = probs.data.tolist()
        sorted_idx = sorted(range(len(prob_list)), key=lambda i: prob_list[i], reverse=True)
        for i in sorted_idx:
            st.progress(prob_list[i], text=f"{names[i]}: {prob_list[i]:.1%}")

    # ── Grad-CAM row ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Grad-CAM Explainability")
    st.write("The heatmap highlights the image regions most influential for the prediction.")

    with st.spinner("Generating Grad-CAM heatmap..."):
        try:
            cam_img = compute_gradcam(image, top1)
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.image(image.resize((224, 224)), caption="Original (224×224)", use_container_width=True)
            with c2:
                st.image(cam_img, caption="Grad-CAM Heatmap", use_container_width=True)
            with c3:
                st.markdown("**How to read this:**")
                st.markdown("- 🔴 **Red/warm** regions: high importance")
                st.markdown("- 🔵 **Blue/cool** regions: low importance")
                st.markdown(f"- Model focused on these areas to predict **{names[top1]}**")

            # ── Claude AI explanation ─────────────────────────────────────────
            st.markdown("---")
            st.markdown("### AI Clinical Explanation")
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                with st.spinner("Generating clinical explanation..."):
                    try:
                        def img_to_b64(img):
                            buf = io.BytesIO()
                            img.save(buf, format="JPEG")
                            return base64.b64encode(buf.getvalue()).decode()

                        client = anthropic.Anthropic(api_key=api_key)
                        response = client.messages.create(
                            model="claude-haiku-4-5-20251001",
                            max_tokens=300,
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_to_b64(image.resize((224, 224)))}},
                                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_to_b64(cam_img)}},
                                    {"type": "text", "text": (
                                        f"You are a clinical AI assistant helping interpret retinal fundus images. "
                                        f"The deep learning model predicted this image as '{names[top1]}' with {top1_conf:.1%} confidence. "
                                        f"The second image is the Grad-CAM heatmap showing which regions most influenced the prediction (red/warm = high importance, blue/cool = low importance). "
                                        f"In 3-4 sentences, explain: (1) what clinical features of {names[top1]} are visible or expected in the fundus image, "
                                        f"(2) whether the heatmap activation regions appear clinically reasonable for this condition. "
                                        f"Be concise and clinically informative."
                                    )}
                                ]
                            }]
                        )
                        st.info(response.content[0].text)
                    except Exception as ex:
                        st.warning(f"Explanation unavailable: {ex}")
            else:
                st.caption("(AI explanation unavailable — API key not configured)")

        except Exception as e:
            st.warning(f"Grad-CAM could not be generated: {e}")
