import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

class ModelVisualizer:
    """
    Visualization utilities for model analysis.
    """
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: str = None):
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot MAE
        ax2.plot(history['mae'], label='Training MAE', linewidth=2)
        ax2.plot(history['val_mae'], label='Validation MAE', linewidth=2)
        ax2.set_title('Model MAE', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 save_path: str = None):
        """
        Plot predictions vs actual values.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, s=30, color='blue')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Price (€)', fontsize=12)
        plt.ylabel('Predicted Price (€)', fontsize=12)
        plt.title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² annotation
        r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
        plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Predictions plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
        """
        Plot residuals analysis.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6, s=30)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax2.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add normal distribution curve
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax2.plot(x, len(residuals) * (1/(sigma * np.sqrt(2 * np.pi))) * 
                np.exp(-0.5 * ((x - mu) / sigma) ** 2), 
                'r-', linewidth=2, label='Normal')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Residuals plot saved to {save_path}")
        
        plt.show()

class ResultsManager:
    """
    Utilities for saving and loading results.
    """
    
    @staticmethod
    def save_results(results: Dict[str, Any], filename: str = "model_results.json"):
        """
        Save model results to JSON file.
        
        Args:
            results: Dictionary of results
            filename: Output filename
        """
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            print(f"📄 Results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    @staticmethod
    def load_results(filename: str = "model_results.json") -> Dict[str, Any]:
        """
        Load results from JSON file.
        
        Args:
            filename: Input filename
            
        Returns:
            Dict[str, Any]: Loaded results
        """
        try:
            with open(filename, 'r') as f:
                results = json.load(f)
            print(f"📄 Results loaded from {filename}")
            return results
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return {}
    
    @staticmethod
    def generate_report(results: Dict[str, Any], model_type: str = "ANN") -> str:
        """
        Generate a comprehensive report.
        
        Args:
            results: Dictionary of results
            model_type: Type of model used
            
        Returns:
            str: Generated report
        """
        report = []
        report.append("=" * 60)
        report.append(f"LAPTOP PRICE PREDICTION - {model_type} MODEL REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Model Performance
        report.append("MODEL PERFORMANCE:")
        report.append("-" * 30)
        report.append(f"R² Score: {results.get('r2_score', 'N/A'):.4f}")
        report.append(f"RMSE: {results.get('rmse', 'N/A'):.2f}")
        report.append(f"MAE: {results.get('mae', 'N/A'):.2f}")
        report.append(f"MSE: {results.get('mse', 'N/A'):.2f}")
        report.append("")
        
        # Interpretation
        r2 = results.get('r2_score', 0)
        if r2 > 0.8:
            interpretation = "Excellent model performance"
        elif r2 > 0.6:
            interpretation = "Good model performance"
        elif r2 > 0.4:
            interpretation = "Moderate model performance"
        else:
            interpretation = "Poor model performance"
        
        report.append(f"MODEL INTERPRETATION: {interpretation}")
        report.append("")
        
        # Business Insights
        report.append("BUSINESS INSIGHTS:")
        report.append("-" * 20)
        report.append("• Model can predict laptop prices with reasonable accuracy")
        report.append("• Helps in competitive pricing analysis")
        report.append("• Useful for market trend analysis")
        report.append("• Can assist in inventory and pricing decisions")
        report.append("")
        
        # Technical Details
        report.append("TECHNICAL DETAILS:")
        report.append("-" * 20)
        report.append("• Architecture: Artificial Neural Network")
        report.append("• Features: Engineered with interactions")
        report.append("• Preprocessing: Comprehensive data cleaning")
        report.append("• Evaluation: Multiple metrics considered")
        
        return "\n".join(report)

def create_sample_predictions(model, feature_engineer, num_samples: int = 5):
    """
    Create sample predictions for demonstration.
    
    Args:
        model: Trained model
        feature_engineer: Feature engineer instance
        num_samples: Number of sample predictions
    """
    print(f"\n🔮 Creating {num_samples} sample predictions...")
    
    # Generate random sample data (this would be replaced with actual sample laptops)
    sample_data = np.random.randn(num_samples, model.input_dim)
    
    # Make predictions
    predictions = model.predict(sample_data)
    
    print("Sample Predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"  Sample {i}: €{pred[0]:.2f}")
