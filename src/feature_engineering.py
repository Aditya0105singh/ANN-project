import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Tuple

class FeatureEngineer:
    """
    Handle feature engineering for laptop price prediction.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_columns = []
        self.numerical_columns = []
    
    def identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify categorical and numerical columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[List[str], List[str]]: (categorical_columns, numerical_columns)
        """
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove target variable from numerical columns if present
        if 'Price_euros' in numerical_cols:
            numerical_cols.remove('Price_euros')
        
        self.categorical_columns = categorical_cols
        self.numerical_columns = numerical_cols
        
        print(f"📊 Identified {len(categorical_cols)} categorical columns")
        print(f"📊 Identified {len(numerical_cols)} numerical columns")
        
        return categorical_cols, numerical_cols
    
    def encode_categorical_variables(self, df: pd.DataFrame, method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): 'onehot' or 'label'
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        df_encoded = df.copy()
        
        if method == 'onehot':
            # One-hot encoding
            df_encoded = pd.get_dummies(df_encoded, columns=self.categorical_columns, drop_first=True)
            df_encoded = df_encoded.astype(int)
            print("✅ Applied one-hot encoding to categorical variables")
            
        elif method == 'label':
            # Label encoding
            for col in self.categorical_columns:
                if col in df_encoded.columns:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
                    self.label_encoders[col] = le
            print("✅ Applied label encoding to categorical variables")
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit the scaler
            
        Returns:
            pd.DataFrame: Dataframe with scaled numerical features
        """
        df_scaled = df.copy()
        
        if fit:
            df_scaled[self.numerical_columns] = self.scaler.fit_transform(df_scaled[self.numerical_columns])
            print("✅ Fitted and scaled numerical features")
        else:
            df_scaled[self.numerical_columns] = self.scaler.transform(df_scaled[self.numerical_columns])
            print("✅ Scaled numerical features using existing scaler")
        
        return df_scaled
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features for better model performance.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with interaction features
        """
        df_interactions = df.copy()
        
        # Create RAM * CPU_Speed interaction
        if 'Ram' in df_interactions.columns and 'CPU_Speed' in df_interactions.columns:
            df_interactions['RAM_CPU_Interaction'] = df_interactions['Ram'] * df_interactions['CPU_Speed']
        
        # Create Inches * Weight interaction (screen size density)
        if 'Inches' in df_interactions.columns and 'Weight' in df_interactions.columns:
            df_interactions['Screen_Weight_Ratio'] = df_interactions['Inches'] / df_interactions['Weight']
        
        # Create Memory_Capacity_GB * Ram interaction
        if 'Memory_Capacity_GB' in df_interactions.columns and 'Ram' in df_interactions.columns:
            df_interactions['Memory_RAM_Ratio'] = df_interactions['Memory_Capacity_GB'] / df_interactions['Ram']
        
        print("✅ Created interaction features")
        return df_interactions
    
    def engineer_features(self, df: pd.DataFrame, encoding_method: str = 'onehot', 
                        create_interactions: bool = True) -> pd.DataFrame:
        """
        Apply complete feature engineering pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            encoding_method (str): 'onehot' or 'label'
            create_interactions (bool): Whether to create interaction features
            
        Returns:
            pd.DataFrame: Fully engineered features
        """
        print("⚙️ Starting feature engineering...")
        
        # Identify column types
        self.identify_column_types(df)
        
        # Encode categorical variables
        df_engineered = self.encode_categorical_variables(df, method=encoding_method)
        
        # Scale numerical features
        df_engineered = self.scale_numerical_features(df_engineered)
        
        # Create interaction features
        if create_interactions:
            df_engineered = self.create_interaction_features(df_engineered)
        
        print(f"✅ Feature engineering completed. Final shape: {df_engineered.shape}")
        return df_engineered
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> dict:
        """
        Get information about feature engineering for documentation.
        
        Returns:
            dict: Feature engineering summary
        """
        return {
            'original_categorical_columns': self.categorical_columns,
            'original_numerical_columns': self.numerical_columns,
            'label_encoders': len(self.label_encoders),
            'scaler_fitted': hasattr(self.scaler, 'mean_')
        }
