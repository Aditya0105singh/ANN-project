import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ----------------------------------
# Streamlit UI - Load model only when needed
# ----------------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

st.title("💻 Laptop Price Prediction")
st.write("Enter laptop specifications to estimate its price.")

# Check if model files exist
if not os.path.exists("laptop_price_model.pkl"):
    st.error("❌ Model file not found!")
    st.warning("Please run `python train_model_simple.py` first to train and save the model.")
    st.stop()

if not os.path.exists("model_columns.pkl") or not os.path.exists("dropdowns.pkl") or not os.path.exists("scaler_X.pkl"):
    st.error("❌ Model metadata files not found!")
    st.warning("Please run `python train_model_simple.py` first to train and save the model.")
    st.stop()

# Load dropdown options
try:
    with open("dropdowns.pkl", "rb") as f:
        dropdowns = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading dropdown options: {e}")
    st.stop()

# ----------------------------------
# User Inputs
# ----------------------------------
st.subheader("🔧 Laptop Specifications")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Brand", dropdowns["Company"])
    type_name = st.selectbox("Laptop Type", dropdowns["TypeName"])
    cpu = st.selectbox("CPU Brand", dropdowns["Cpu_brand"])
    gpu = st.selectbox("GPU Brand", dropdowns["Gpu_brand"])
    os = st.selectbox("Operating System", dropdowns["OpSys"])

with col2:
    ram = st.selectbox("RAM (GB)", dropdowns["Ram"])
    inches = st.number_input("Screen Size (Inches)", 10.0, 20.0, step=0.1, value=15.6)
    ssd = st.number_input("SSD (GB)", 0, 2000, step=128, value=256)
    hdd = st.number_input("HDD (GB)", 0, 2000, step=256, value=0)
    weight = st.number_input("Weight (kg)", 0.5, 5.0, step=0.1, value=2.0)

# ----------------------------------
# Predict Button - Load model only when prediction is needed
# ----------------------------------
if st.button("🔮 Predict Price"):
    with st.spinner("🔄 Loading model and making prediction..."):
        try:
            # Load model and metadata only when needed
            with open("laptop_price_model.pkl", "rb") as f:
                model = pickle.load(f)
            
            with open("model_columns.pkl", "rb") as f:
                model_columns = pickle.load(f)
            
            with open("scaler_X.pkl", "rb") as f:
                scaler_X = pickle.load(f)
            
            # Create single-row dataframe
            input_data = {
                "Company": company,
                "TypeName": type_name,
                "Cpu_brand": cpu,
                "Gpu_brand": gpu,
                "OpSys": os,
                "Ram": ram,
                "Inches": inches,
                "SSD": ssd,
                "HDD": hdd,
                "Weight": weight
            }

            input_df = pd.DataFrame([input_data])

            # One-hot encode
            encoded_df = pd.get_dummies(input_df)

            # Align columns with training data
            encoded_df = encoded_df.reindex(columns=model_columns, fill_value=0)

            # Scale numerical features
            numerical_features = ['Ram', 'Inches', 'Weight', 'SSD', 'HDD']
            encoded_df[numerical_features] = scaler_X.transform(encoded_df[numerical_features])

            # Prediction
            prediction = model.predict(encoded_df)[0]

            st.success(f"💰 Estimated Laptop Price: ₹{int(prediction):,}")
            
            # Add confidence indicator based on typical ranges
            if prediction < 30000:
                st.info("💡 This appears to be a budget laptop")
            elif prediction < 80000:
                st.info("💡 This appears to be a mid-range laptop")
            elif prediction < 150000:
                st.info("💡 This appears to be a premium laptop")
            else:
                st.info("💡 This appears to be a high-end/gaming laptop")

            st.caption("⚠️ Prediction is based on historical data and may vary. Actual prices may differ based on brand, market conditions, and specific configurations.")
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.write("Please check that all model files exist and are valid.")

# ----------------------------------
# Footer and Additional Info
# ----------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Machine Learning")

# Show model info
with st.expander("📊 Model Information"):
    st.write("**Model Type:** Random Forest Regressor")
    st.write("**Dataset:** 20 laptop samples with prices")
    st.write("**Features:** Brand, Type, CPU, GPU, OS, RAM, Screen Size, Storage, Weight")
    st.write("**Performance:** MAE ≈ ₹12,000, R² ≈ 0.65")
    st.write("**Output:** Estimated price in Indian Rupees (₹)")
    
    st.write("**Price Ranges in Dataset:**")
    st.write("- Minimum: ₹20,610")
    st.write("- Maximum: ₹228,370") 
    st.write("- Average: ₹126,660")

# Show some example predictions
with st.expander("🔍 Example Predictions"):
    st.write("**Sample laptop configurations and their predicted prices:**")
    
    examples = [
        {"Brand": "Dell", "Type": "Ultrabook", "RAM": 16, "SSD": 256, "Price": "~₹1,17,000"},
        {"Brand": "Apple", "Type": "Ultrabook", "RAM": 8, "SSD": 128, "Price": "~₹1,20,000"},
        {"Brand": "HP", "Type": "Notebook", "RAM": 8, "SSD": 256, "Price": "~₹52,000"},
        {"Brand": "Acer", "Type": "Ultrabook", "RAM": 8, "SSD": 128, "Price": "~₹27,000"}
    ]
    
    for i, example in enumerate(examples, 1):
        st.write(f"{i}. {example['Brand']} {example['Type']} ({example['RAM']}GB RAM, {example['SSD']}GB SSD) - {example['Price']}")
