# Laptop Price Prediction using Artificial Neural Networks

This project implements a sophisticated Artificial Neural Network (ANN) to predict laptop prices based on various hardware and software specifications.

## 🎯 Project Overview

A comprehensive machine learning pipeline that predicts laptop prices using deep learning techniques with advanced feature engineering and data preprocessing.

## 📊 Dataset

- **Source**: Laptop price dataset with 1303 records
- **Target Variable**: Price_euros (laptop price in Euros)
- **Features**: Company, Product, Type, Screen specs, CPU, RAM, Memory, GPU, OS, Weight, etc.

## 🏗️ Architecture

### Neural Network Models
- **Standard Architecture**: 3-layer ANN with batch normalization
- **Deep Architecture**: 5-layer ANN for complex patterns
- **Wide Architecture**: Wide layers for high-dimensional data

### Key Components
- **Data Loading**: Robust CSV loading with comprehensive analysis
- **Preprocessing**: Advanced cleaning and feature extraction
- **Feature Engineering**: One-hot encoding, scaling, interaction features
- **Model Training**: Multiple ANN architectures with callbacks
- **Evaluation**: Comprehensive metrics and visualization

## 🚀 Features

### Data Processing
- ✅ Automatic data type detection and conversion
- ✅ Missing value handling and outlier detection
- ✅ Feature extraction from complex strings (CPU, Memory, Screen)
- ✅ Interaction feature creation

### Model Capabilities
- ✅ Multiple ANN architectures
- ✅ Early stopping and learning rate scheduling
- ✅ Batch normalization and dropout regularization
- ✅ Comprehensive evaluation metrics

### Visualization & Analysis
- ✅ Training history plots
- ✅ Prediction vs actual comparisons
- ✅ Residual analysis
- ✅ Results management and reporting

## 📁 Project Structure

```
laptop-price-ann/
│
├── data/
│   └── laptop_price.csv
│
├── src/
│   ├── data_loader.py       # Data loading and exploration
│   ├── data_preprocessor.py # Data cleaning and preprocessing
│   ├── feature_engineering.py # Feature engineering pipeline
│   ├── ann_model.py         # Neural network architectures
│   ├── train.py             # Complete training pipeline
│   └── utils.py             # Visualization and utilities
│
├── requirements.txt
├── README.md
└── ann project.ipynb        # Original notebook
```

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

### Quick Start
```bash
cd src
python train.py
```

### Advanced Usage
```python
from train import LaptopPriceTrainer

# Initialize trainer
trainer = LaptopPriceTrainer("data/laptop_price.csv")

# Run complete pipeline
results = trainer.run_complete_pipeline(architecture='deep')
```

## 📈 Model Performance

The ANN model achieves competitive performance in laptop price prediction with:
- **R² Score**: Measures model fit
- **RMSE**: Root Mean Square Error in price prediction
- **MAE**: Mean Absolute Error for price estimates
- **Training History**: Loss and MAE tracking over epochs

## 🔧 Technical Highlights

### Advanced Preprocessing
- **Screen Resolution**: Extract resolution, IPS panel, touchscreen detection
- **CPU Features**: Brand, family, and speed extraction
- **Memory Analysis**: Capacity and type identification
- **Weight/RAM**: Unit conversion and normalization

### Feature Engineering
- **One-Hot Encoding**: For categorical variables
- **Standard Scaling**: For numerical features
- **Interaction Features**: RAM×CPU, Screen×Weight, Memory×RAM ratios

### Neural Network Design
- **Batch Normalization**: Stabilizes training
- **Dropout Regularization**: Prevents overfitting
- **Early Stopping**: Prevents overtraining
- **Learning Rate Scheduling**: Adaptive optimization

## 📊 Business Applications

- **Price Optimization**: Competitive pricing analysis
- **Market Analysis**: Price trend identification
- **Inventory Management**: Stock pricing decisions
- **Customer Insights**: Price sensitivity analysis

## 🔮 Future Improvements

- [ ] Hyperparameter optimization with GridSearch/RandomSearch
- [ ] Ensemble methods combining multiple models
- [ ] Advanced feature selection techniques
- [ ] Real-time prediction API deployment
- [ ] Cross-validation for robust evaluation

## 📝 Development Notes

This project demonstrates:
- **End-to-end ML pipeline** development
- **Deep learning** for regression tasks
- **Feature engineering** best practices
- **Model evaluation** and visualization
- **Clean code architecture** and documentation

## 🤝 Contributing

Feel free to contribute improvements, bug fixes, or new features!

## 📄 License

This project is for educational and demonstration purposes.
