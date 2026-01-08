import streamlit as st
import requests
from PIL import Image
import io
import time

st.set_page_config(
    page_title="BISINDO AI Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000/predict"

st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            height: 3em;
            background-color: #FF4B4B;
            color: white;
            font-weight: bold;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2.5rem;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Settings")
    
    input_method = st.selectbox(
        "Select Input Method:",
        ("Upload Image 📁", "Webcam 📷")
    )
    
    st.markdown("---")
    st.markdown("### About the Model")
    st.info(
        """
        This model uses the **ResNet18** architecture, fine-tuned 
        to recognize 26 alphabets of **BISINDO** (Indonesian Sign Language).
        """
    )

st.title("🤟 BISINDO Sign Language Prediction")
st.markdown("### Translate hand signs into letter.")

main_container = st.container()

image_file = None

if input_method == "Upload Image 📁":
    image_file = st.file_uploader("Upload a hand image (Clear & Bright)", type=["jpg", "jpeg", "png"])
else:
    image_file = st.camera_input("Capture hand sign")

if image_file is not None:
    image = Image.open(image_file)
    
    with main_container:
        st.write("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Image")
            st.image(image, caption="Image to be processed", width=300)
        
        with col2:
            st.subheader("Prediction Result")
            
            predict_btn = st.button("🔍 Analyze Image")
            
            if predict_btn:
                progress_text = "Processing image..."
                my_bar = st.progress(0, text=progress_text)
                
                try:
                    for percent_complete in range(0, 100, 20):
                        time.sleep(0.03)
                        my_bar.progress(percent_complete + 10, text=progress_text)

                    img_bytes = io.BytesIO()
                    save_format = image.format if image.format else "JPEG"
                    image.save(img_bytes, format=save_format)
                    img_bytes = img_bytes.getvalue()

                    filename = image_file.name if hasattr(image_file, 'name') else "camera_capture.jpg"
                    filetype = image_file.type if hasattr(image_file, 'type') else "image/jpeg"

                    files = {"file": (filename, img_bytes, filetype)}
                    
                    response = requests.post(API_URL, files=files)
                    my_bar.empty() # Remove loading bar

                    if response.status_code == 200:
                        result = response.json()
                        label = result["predicted_label"]
                        confidence = result["confidence"]

                        st.success("Analysis Complete!")
                        
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("Predicted Letter", f"{label}")
                        with metric_col2:
                            st.metric("Confidence", f"{confidence:.1%}")
                        
                        st.caption("Model Confidence Level:")
                        st.progress(confidence)

                        if confidence > 0.9:
                            st.balloons()
                            st.markdown(f"**Great!** The model is very confident this is **{label}**.")
                        elif confidence < 0.5:
                            st.warning("The model is unsure. Try improving lighting or hand position.")
                            
                    else:
                        st.error(f"Server Error: {response.status_code}")
                        with st.expander("View Error Details"):
                            st.write(response.text)

                except requests.exceptions.ConnectionError:
                    my_bar.empty()
                    st.error("❌ Failed to connect to FastAPI Server.")
                    st.info("Make sure `app.py` is running: `uvicorn app:app --reload`")
                except Exception as e:
                    my_bar.empty()
                    st.error(f"Internal error occurred: {e}")

else:
    st.info("👈 Please select an input method from the sidebar and upload an image to start.")

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        <p style="font-size: 12px;">© 2026 abidalfrz. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)