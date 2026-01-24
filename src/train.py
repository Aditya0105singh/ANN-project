import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from data_loader import LaptopDataLoader
from data_preprocessor import LaptopDataPreprocessor
from feature_engineering import FeatureEngineer
from ann_model import LaptopPriceANN

class LaptopPriceTrainer:
    """
    Complete training pipeline for laptop price prediction.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.loader = LaptopDataLoader(data_path)
        self.preprocessor = LaptopDataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
    
    def load_and_explore_data(self) -> pd.DataFrame:
        """Load and explore the dataset."""
        print("🔍 Loading and exploring data...")
        
        self.data = self.loader.load_data()
        if self.data is None:
            raise ValueError("Failed to load data")
        
        self.loader.get_basic_info()
        self.loader.get_statistical_summary()
        self.loader.get_categorical_info()
        
        return self.data
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the dataset."""
        print("\n🧹 Preprocessing data...")
        
        processed_data = self.preprocessor.preprocess_all(self.data)
        print(f"✅ Preprocessing completed. Shape: {processed_data.shape}")
        
        return processed_data
    
    def engineer_features(self) -> pd.DataFrame:
        """Apply feature engineering."""
        print("\n⚙️ Engineering features...")
        
        engineered_data = self.feature_engineer.engineer_features(
            self.preprocessor.processed_data,
            encoding_method='onehot',
            create_interactions=True
        )
        
        return engineered_data
    
    def prepare_train_test_split(self, engineered_data: pd.DataFrame) -> None:
        """Prepare train/validation/test splits."""
        print("\n📦 Preparing train/test splits...")
        
        # Separate features and target
        X = engineered_data.drop('Price_euros', axis=1)
        y = engineered_data['Price_euros']
        
        # Split into train+validation and test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Split train+validation into train and validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42
        )
        
        print(f"✅ Data splits prepared:")
        print(f"  Training: {self.X_train.shape[0]} samples")
        print(f"  Validation: {self.X_val.shape[0]} samples")
        print(f"  Test: {self.X_test.shape[0]} samples")
        print(f"  Features: {self.X_train.shape[1]}")
    
    def build_and_train_model(self, architecture: str = 'standard') -> dict:
        """Build and train the neural network."""
        print(f"\n🤖 Building and training {architecture} ANN...")
        
        # Initialize model
        self.model = LaptopPriceANN(input_dim=self.X_train.shape[1])
        
        # Build model
        self.model.build_model(architecture=architecture)
        
        # Print model summary
        print("\n📋 Model Architecture:")
        print(self.model.get_model_summary())
        
        # Train model
        history = self.model.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            epochs=100,
            batch_size=32,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self) -> dict:
        """Evaluate the trained model."""
        print("\n📊 Evaluating model performance...")
        
        # Evaluate on test set
        test_metrics = self.model.evaluate(self.X_test, self.y_test)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate additional metrics
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        
        evaluation_results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'test_loss': test_metrics['loss'],
            'test_mae': test_metrics['mae']
        }
        
        print(f"\n🎯 Final Evaluation Results:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  MSE: {mse:.2f}")
        
        return evaluation_results
    
    def run_complete_pipeline(self, architecture: str = 'standard') -> dict:
        """Run the complete training pipeline."""
        print("🚀 Starting Complete Laptop Price Prediction Pipeline")
        print("=" * 60)
        
        try:
            # Load and explore data
            self.load_and_explore_data()
            
            # Preprocess data
            self.preprocess_data()
            
            # Engineer features
            engineered_data = self.engineer_features()
            
            # Prepare splits
            self.prepare_train_test_split(engineered_data)
            
            # Build and train model
            history = self.build_and_train_model(architecture)
            
            # Evaluate model
            results = self.evaluate_model()
            
            print("\n🎉 Pipeline completed successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            raise

def main():
    """Main function to run the training pipeline."""
    # Initialize trainer
    trainer = LaptopPriceTrainer("data/laptop_price.csv")
    
    # Run complete pipeline
    results = trainer.run_complete_pipeline(architecture='standard')
    
    return trainer, results

if __name__ == "__main__":
    trainer, results = main()
