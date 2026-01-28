# 💻 Laptop Price Predictor

A machine learning application that predicts laptop prices based on specifications.

## 🚀 **Live Demo**

**🔗 [https://aditya0105singh-ann-project.streamlit.app/](https://aditya0105singh-ann-project.streamlit.app/)

*Click the link above to try the live application!*

## 📊 Model Performance

- **MAE**: ₹12,331
- **R² Score**: 0.653  
- **Model**: RandomForest Regressor
- **Features**: Brand, Type, CPU, GPU, RAM, Storage, Screen Size, Weight

## � Features

- ✅ **Accurate Predictions**: Based on real laptop data
- ✅ **Interactive UI**: Streamlit-based web interface
- ✅ **Real-time**: Instant price estimates
- ✅ **Mobile Responsive**: Works on all devices
- ✅ **Smart Categories**: Budget/Mid-range/Premium indicators

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, Scikit-learn
- **Model**: RandomForest Regressor
- **Deployment**: Streamlit Cloud

## 📁 Project Structure

```
├── app_final.py              # Main Streamlit application
├── train_model_simple.py     # Model training script
├── laptop_price_model.pkl    # Trained model
├── model_columns.pkl         # Feature columns
├── dropdowns.pkl             # UI dropdown options
├── scaler_X.pkl              # Feature scaler
├── requirements_deploy.txt   # Dependencies
├── Dockerfile               # Docker configuration
└── data/                    # Dataset folder
```

## � Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aditya0105singh/ANN-project.git
   cd ANN-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_deploy.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app_final.py
   ```

4. **Open in browser**: `http://localhost:8501`

## 📖 Deployment

### **Streamlit Cloud (Live)**
- **URL**: https://aditya0105singh-ann-project.streamlit.app/
- **Platform**: Streamlit Community Cloud
- **Status**: ✅ Deployed and Live

### **Other Deployment Options**
- **Docker**: `docker build -t laptop-predictor . && docker run -p 8501:8501 laptop-predictor`
- **Heroku**: Free tier with Procfile
- **PythonAnywhere**: Manual upload

## 📊 Sample Predictions

- **Dell Ultrabook** (16GB RAM, 256GB SSD): ~₹1,17,000
- **Apple MacBook** (8GB RAM, 128GB SSD): ~₹1,20,000  
- **HP Notebook** (8GB RAM, 256GB SSD): ~₹52,000

## 🤖 Model Training

To retrain the model:
```bash
python train_model_simple.py
```

## 📈 Key Features

### **Data Processing**
- ✅ CPU/GPU brand extraction
- ✅ Storage type detection (SSD/HDD)
- ✅ Memory and weight normalization
- ✅ One-hot encoding for categorical variables

### **Model Performance**
- ✅ Mean Absolute Error: ₹12,331
- ✅ R² Score: 0.653
- ✅ Feature importance analysis
- ✅ Cross-validation

### **User Interface**
- ✅ Intuitive dropdown menus
- ✅ Real-time price prediction
- ✅ Confidence indicators
- ✅ Error handling
- ✅ Mobile-friendly design

## 🔧 Technical Implementation

### **Feature Engineering**
```python
# CPU Brand Extraction
def extract_cpu_brand(cpu_str):
    if 'Intel' in cpu_str:
        return 'Intel'
    elif 'AMD' in cpu_str:
        return 'AMD'
    else:
        return 'Other'

# GPU Brand Extraction  
def extract_gpu_brand(gpu_str):
    if 'Nvidia' in gpu_str or 'GeForce' in gpu_str:
        return 'Nvidia'
    elif 'Intel' in gpu_str:
        return 'Intel'
    elif 'AMD' in gpu_str:
        return 'AMD'
    else:
        return 'Other'
```

### **Model Training**
```python
# RandomForest with optimized parameters
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

## 🌟 Business Applications

- **Price Optimization**: Competitive pricing analysis
- **Market Analysis**: Price trend identification  
- **Inventory Management**: Stock pricing decisions
- **Customer Insights**: Price sensitivity analysis

## 📝 Development Process

1. **Data Collection**: Laptop specifications dataset
2. **Preprocessing**: Feature extraction and cleaning
3. **Model Selection**: RandomForest for best performance
4. **Training**: Cross-validation and hyperparameter tuning
5. **Deployment**: Streamlit web application
6. **Testing**: Real-world validation

## 🤝 Contributing

Feel free to contribute improvements, bug fixes, or new features!

## 📄 License

This project is for educational and demonstration purposes.
