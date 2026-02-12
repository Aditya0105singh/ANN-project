import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sklearn

# ----------------------------------
# Streamlit UI Configuration
# ----------------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    h1 {
        color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Sidebar & Header
# ----------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/428/428001.png", width=100)
    st.title("Price Predictor")
    st.write("Using Machine Learning to estimate laptop prices based on specifications.")
    st.markdown("---")
    st.info(f"Scikit-learn version: {sklearn.__version__}")
    st.markdown("---")
    st.write("Created by Aditya Singh")

col1, col2 = st.columns([3, 1])
with col1:
    st.title("💻 Laptop Price Estimator")
    st.markdown("Use the controls below to configure a laptop and get an instant price prediction.")
with col2:
    # Placeholder for potential logo or extra metric
    pass

st.markdown("---")

# ----------------------------------
# Load Model Artifacts
# ----------------------------------
model_files = ["laptop_price_model.pkl", "model_columns.pkl", "dropdowns.pkl", "scaler_X.pkl"]
missing_files = [f for f in model_files if not os.path.exists(f)]

if missing_files:
    st.error("❌ Model files not found!")
    st.warning(f"Missing: {', '.join(missing_files)}")
    st.info("Please run `train_model_simple.py` to regenerate the model.")
    st.stop()

try:
    with open("dropdowns.pkl", "rb") as f:
        dropdowns = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading configuration: {e}")
    st.stop()

# ----------------------------------
# User Inputs (Grid Layout)
# ----------------------------------
with st.container():
    st.subheader("🛠️ Configuration")
    
    # Row 1: Core Specs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        company = st.selectbox("🏷️ Brand", dropdowns["Company"])
        type_name = st.selectbox("💻 Type", dropdowns["TypeName"])
        
    with col2:
        cpu = st.selectbox("🧠 CPU Brand", dropdowns["Cpu_brand"])
        gpu = st.selectbox("🎮 GPU Brand", dropdowns["Gpu_brand"])

    with col3:
        os_sys = st.selectbox("⚙️ Operating System", dropdowns["OpSys"])
        ram = st.select_slider("🚀 RAM (GB)", options=dropdowns["Ram"], value=8)

    st.markdown("---")

    # Row 2: Details
    col4, col5 = st.columns(2)
    
    with col4:
        st.write("💾 **Storage Configuration**")
        c1, c2 = st.columns(2)
        with c1:
            ssd = st.slider("SSD Storage (GB)", 0, 4096, 256, step=128)
        with c2:
            hdd = st.slider("HDD Storage (GB)", 0, 2048, 0, step=256)
            
    with col5:
        st.write("🖥️ **Display & Build**")
        c3, c4 = st.columns(2)
        with c3:
            inches = st.slider("Screen Size (Inches)", 10.0, 18.0, 15.6, step=0.1)
        with c4:
            weight = st.slider("Weight (kg)", 0.5, 5.0, 2.0, step=0.1)

# ----------------------------------
# Prediction Logic
# ----------------------------------
st.markdown("---")
if st.button("🔮 Predict Price", type="primary"):
    
    # Progress bar effect
    progress_text = "Analyzing specifications..."
    my_bar = st.progress(0, text=progress_text)

    try:
        # Load model on demand
        with open("laptop_price_model.pkl", "rb") as f:
            model = pickle.load(f)
        my_bar.progress(30, text="Loading AI model...")
        
        with open("model_columns.pkl", "rb") as f:
            model_columns = pickle.load(f)
        my_bar.progress(50, text="Processing features...")
        
        with open("scaler_X.pkl", "rb") as f:
            scaler_X = pickle.load(f)
        
        # Create input dataframe
        input_data = {
            "Company": company,
            "TypeName": type_name,
            "Cpu_brand": cpu,
            "Gpu_brand": gpu,
            "OpSys": os_sys,
            "Ram": ram,
            "Inches": inches,
            "SSD": ssd,
            "HDD": hdd,
            "Weight": weight
        }

        input_df = pd.DataFrame([input_data])
        encoded_df = pd.get_dummies(input_df)
        encoded_df = encoded_df.reindex(columns=model_columns, fill_value=0)
        
        my_bar.progress(80, text="Scaling data...")
        numerical_features = ['Ram', 'Inches', 'Weight', 'SSD', 'HDD']
        encoded_df[numerical_features] = scaler_X.transform(encoded_df[numerical_features])

        # Predict
        prediction = model.predict(encoded_df)[0]
        my_bar.progress(100, text="Done!")
        my_bar.empty()

        # ----------------------------------
        # Result Display
        # ----------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        res_col1, res_col2, res_col3 = st.columns([1,2,1])
        
        with res_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color:#555;">Estimated Price</h3>
                <h1 style="margin:0; color:#00cc66; font-size: 3rem;">₹{int(prediction):,}</h1>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Contextual Info
        if prediction < 40000:
            st.info("💡 **Budget Friendly**: Great for basic tasks, browsing, and students.")
        elif prediction < 80000:
            st.success("💡 **Mid-Range**: Good for multitasking, light gaming, and office work.")
        elif prediction < 150000:
            st.warning("💡 **Premium**: Excellent build quality, high performance for professionals.")
        else:
            st.error("💡 **High-End / Gaming**: Top-tier performance for heavy gaming or video editing.")
            
    except Exception as e:
        my_bar.empty()
        st.error(f"❌ Prediction failed: {str(e)}")
        st.expander("Show Error Details").write(e)
