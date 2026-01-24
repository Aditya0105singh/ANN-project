import pandas as pd
import numpy as np
from typing import Optional, Tuple

class LaptopDataLoader:
    """
    Utility class for loading and basic exploration of laptop price dataset.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
    
    def load_data(self, encoding: str = 'latin1') -> Optional[pd.DataFrame]:
        """
        Load the laptop price dataset.
        
        Args:
            encoding (str): File encoding, default 'latin1'
            
        Returns:
            pd.DataFrame: Loaded dataset or None if failed
        """
        try:
            self.data = pd.read_csv(self.file_path, encoding=encoding)
            print(f"✅ Dataset loaded successfully: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"❌ Error: File {self.file_path} not found")
            return None
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def get_basic_info(self) -> None:
        """Display basic information about the dataset."""
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n" + "="*50)
        print("DATASET BASIC INFORMATION")
        print("="*50)
        print(f"Shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print(f"\nData types:\n{self.data.dtypes}")
        print(f"\nMissing values:\n{self.data.isnull().sum()}")
        print(f"\nDuplicate rows: {self.data.duplicated().sum()}")
    
    def get_statistical_summary(self) -> None:
        """Display statistical summary of numerical columns."""
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n" + "="*50)
        print("STATISTICAL SUMMARY")
        print("="*50)
        print(self.data.describe())
    
    def get_categorical_info(self) -> None:
        """Display information about categorical columns."""
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        print("\n" + "="*50)
        print("CATEGORICAL COLUMNS INFO")
        print("="*50)
        
        for col in categorical_cols:
            print(f"\n{col}:")
            print(f"  Unique values: {self.data[col].nunique()}")
            print(f"  Top 5 values:\n{self.data[col].value_counts().head()}")
