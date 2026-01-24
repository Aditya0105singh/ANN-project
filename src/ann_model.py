import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Dict, Any
import numpy as np

class LaptopPriceANN:
    """
    Artificial Neural Network for laptop price prediction.
    """
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = None
        self.history = None
        self.callbacks = []
    
    def build_model(self, architecture: str = 'standard') -> tf.keras.Model:
        """
        Build the neural network architecture.
        
        Args:
            architecture (str): Type of architecture ('standard', 'deep', 'wide')
            
        Returns:
            tf.keras.Model: Compiled Keras model
        """
        if architecture == 'standard':
            self.model = self._build_standard_model()
        elif architecture == 'deep':
            self.model = self._build_deep_model()
        elif architecture == 'wide':
            self.model = self._build_wide_model()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        print(f"✅ Built {architecture} neural network architecture")
        return self.model
    
    def _build_standard_model(self) -> tf.keras.Model:
        """Build standard 3-layer neural network."""
        model = Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            Dense(1, activation='linear')  # Regression output
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_deep_model(self) -> tf.keras.Model:
        """Build deeper neural network with more layers."""
        model = Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            Dense(16, activation='relu'),
            
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_wide_model(self) -> tf.keras.Model:
        """Build wider neural network with more neurons."""
        model = Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def setup_callbacks(self, patience: int = 10) -> list:
        """
        Setup training callbacks.
        
        Args:
            patience (int): Early stopping patience
            
        Returns:
            list: List of callbacks
        """
        self.callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print("✅ Callbacks configured")
        return self.callbacks
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              verbose: int = 1) -> Dict[str, Any]:
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Dict[str, Any]: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if not self.callbacks:
            self.setup_callbacks()
        
        print(f"🚀 Training neural network for {epochs} epochs...")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks,
            verbose=verbose
        )
        
        print("✅ Training completed")
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = dict(zip(self.model.metrics_names, results))
        
        print(f"📊 Test Results:")
        print(f"  MSE: {metrics['loss']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        
        return metrics
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built yet."
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()
