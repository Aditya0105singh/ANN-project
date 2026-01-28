import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def extract_cpu_brand(cpu_str):
    """Extract CPU brand from CPU string"""
    if 'Intel' in cpu_str:
        return 'Intel'
    elif 'AMD' in cpu_str:
        return 'AMD'
    else:
        return 'Other'

def extract_gpu_brand(gpu_str):
    """Extract GPU brand from GPU string"""
    if 'Nvidia' in gpu_str or 'GeForce' in gpu_str or 'GTX' in gpu_str or 'RTX' in gpu_str:
        return 'Nvidia'
    elif 'Intel' in gpu_str:
        return 'Intel'
    elif 'AMD' in gpu_str or 'Radeon' in gpu_str:
        return 'AMD'
    else:
        return 'Other'

def extract_storage(memory_str):
    """Extract SSD and HDD storage from Memory string"""
    ssd = hdd = 0
    if 'SSD' in memory_str:
        if 'GB' in memory_str:
            ssd = int(memory_str.split('GB')[0])
        elif 'TB' in memory_str:
            ssd = int(float(memory_str.split('TB')[0]) * 1024)
    elif 'HDD' in memory_str:
        if 'GB' in memory_str:
            hdd = int(memory_str.split('GB')[0])
        elif 'TB' in memory_str:
            hdd = int(float(memory_str.split('TB')[0]) * 1024)
    elif 'Flash Storage' in memory_str:
        if 'GB' in memory_str:
            ssd = int(memory_str.split('GB')[0])
    return ssd, hdd

def main():
    print("🚀 Starting simple but effective model training...")
    
    # Load data
    df = pd.read_csv('data/laptop_price.csv', encoding='latin1')
    print(f"📊 Dataset shape: {df.shape}")
    
    # Data cleaning and preprocessing
    print("🧹 Cleaning and preprocessing data...")
    
    # Clean Ram and Weight columns
    if df['Ram'].dtype == 'object':
        df['Ram'] = df['Ram'].str.replace('GB','').astype(int)
    if df['Weight'].dtype == 'object':
        df['Weight'] = df['Weight'].str.replace('kg','').astype('float')
    
    # Extract CPU and GPU brands
    df['Cpu_brand'] = df['Cpu'].apply(extract_cpu_brand)
    df['Gpu_brand'] = df['Gpu'].apply(extract_gpu_brand)
    
    # Extract storage
    df[['SSD', 'HDD']] = df['Memory'].apply(lambda x: pd.Series(extract_storage(x)))
    
    # Create price in INR (1 EUR ≈ 90 INR)
    df['Price_INR'] = df['Price_euros'] * 90
    
    print(f"💰 Price range: ₹{df['Price_INR'].min():,.0f} - ₹{df['Price_INR'].max():,.0f}")
    print(f"💰 Average price: ₹{df['Price_INR'].mean():,.0f}")
    
    # Feature selection - use more realistic features
    features = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'OpSys', 'Ram', 'Inches', 'Weight', 'SSD', 'HDD']
    df_features = df[features].copy()
    
    # Handle categorical variables
    categorical_features = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'OpSys']
    numerical_features = ['Ram', 'Inches', 'Weight', 'SSD', 'HDD']
    
    # Create dropdown options for UI
    dropdowns = {
        'Company': sorted(df['Company'].unique()),
        'TypeName': sorted(df['TypeName'].unique()),
        'Cpu_brand': sorted(df['Cpu_brand'].unique()),
        'Gpu_brand': sorted(df['Gpu_brand'].unique()),
        'OpSys': sorted(df['OpSys'].unique()),
        'Ram': sorted(df['Ram'].unique())
    }
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df_features, columns=categorical_features, drop_first=True)
    
    # Store column names for prediction
    model_columns = df_encoded.columns.tolist()
    
    # Scale numerical features
    scaler_X = StandardScaler()
    df_encoded[numerical_features] = scaler_X.fit_transform(df_encoded[numerical_features])
    
    # Prepare target variable
    y = df['Price_INR'].values
    
    # Train-test split
    X = df_encoded
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"📚 Training data shape: {x_train.shape}")
    print(f"🧪 Test data shape: {x_test.shape}")
    
    # Use RandomForest instead of Neural Network (better for small datasets)
    print("🌳 Training RandomForest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    # Train model
    model.fit(x_train, y_train)
    
    # Evaluate model
    print("📊 Evaluating model performance...")
    y_pred = model.predict(x_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 Model Performance Metrics:")
    print(f"   MAE: ₹{mae:,.0f}")
    print(f"   R² Score: {r2:.3f}")
    
    # Test with sample data
    print("\n🧪 Testing with sample predictions...")
    for i in range(min(3, len(x_test))):
        sample_input = x_test.iloc[[i]]
        sample_actual = y_test[i]
        sample_pred = model.predict(sample_input)[0]
        
        print(f"   Sample {i+1}:")
        print(f"     Actual: ₹{sample_actual:,.0f}")
        print(f"     Predicted: ₹{sample_pred:,.0f}")
        print(f"     Difference: ₹{abs(sample_actual - sample_pred):,.0f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': model_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🎯 Top 5 Most Important Features:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Save model and metadata
    print("💾 Saving model and metadata...")
    
    # Save as pickle for compatibility
    with open("laptop_price_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open("model_columns.pkl", "wb") as f:
        pickle.dump(model_columns, f)
    
    with open("dropdowns.pkl", "wb") as f:
        pickle.dump(dropdowns, f)
    
    with open("scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    
    print("✅ Model training complete!")
    print("📁 Files saved:")
    print("   - laptop_price_model.pkl")
    print("   - model_columns.pkl") 
    print("   - dropdowns.pkl")
    print("   - scaler_X.pkl")

if __name__ == "__main__":
    main()
