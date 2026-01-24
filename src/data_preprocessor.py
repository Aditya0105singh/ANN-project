import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple

class LaptopDataPreprocessor:
    """
    Handle data preprocessing for laptop price prediction.
    """
    
    def __init__(self):
        self.processed_data = None
    
    def clean_ram_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean RAM and Weight columns by removing units and converting to numeric.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Clean RAM column
        if df_clean['Ram'].dtype == 'object':
            df_clean['Ram'] = df_clean['Ram'].str.replace('GB', '').astype(int)
            print("✅ RAM column cleaned and converted to integer")
        
        # Clean Weight column
        if df_clean['Weight'].dtype == 'object':
            df_clean['Weight'] = df_clean['Weight'].str.replace('kg', '').astype(float)
            print("✅ Weight column cleaned and converted to float")
        
        return df_clean
    
    def extract_screen_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract screen resolution and type from ScreenResolution column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with extracted screen features
        """
        df_clean = df.copy()
        
        # Extract resolution
        def extract_resolution(resolution_str):
            match = re.search(r'(\d+)x(\d+)', resolution_str)
            if match:
                return f"{match.group(1)}x{match.group(2)}"
            return "Unknown"
        
        df_clean['Resolution'] = df_clean['ScreenResolution'].apply(extract_resolution)
        
        # Extract screen type (IPS, Touchscreen, etc.)
        df_clean['IPS_Panel'] = df_clean['ScreenResolution'].str.contains('IPS Panel').astype(int)
        df_clean['Touchscreen'] = df_clean['ScreenResolution'].str.contains('Touchscreen').astype(int)
        df_clean['Retina'] = df_clean['ScreenResolution'].str.contains('Retina').astype(int)
        
        print("✅ Screen features extracted")
        return df_clean
    
    def extract_cpu_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract CPU brand, family, and speed from Cpu column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with extracted CPU features
        """
        df_clean = df.copy()
        
        # Extract CPU brand
        df_clean['CPU_Brand'] = df_clean['Cpu'].apply(lambda x: x.split()[0])
        
        # Extract CPU family
        def extract_cpu_family(cpu_str):
            parts = cpu_str.split()
            for i, part in enumerate(parts):
                if part in ['Core', 'Xeon', 'Pentium', 'Celeron', 'Ryzen', 'Athlon']:
                    return f"{part} {parts[i+1]}" if i+1 < len(parts) else part
            return parts[0]
        
        df_clean['CPU_Family'] = df_clean['Cpu'].apply(extract_cpu_family)
        
        # Extract CPU speed
        def extract_cpu_speed(cpu_str):
            match = re.search(r'(\d+\.?\d*)GHz', cpu_str)
            return float(match.group(1)) if match else 0.0
        
        df_clean['CPU_Speed'] = df_clean['Cpu'].apply(extract_cpu_speed)
        
        print("✅ CPU features extracted")
        return df_clean
    
    def clean_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract memory type and capacity from Memory column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with cleaned memory features
        """
        df_clean = df.copy()
        
        # Extract memory capacity
        def extract_memory_capacity(memory_str):
            match = re.search(r'(\d+\.?\d*)\s*(GB|TB)', memory_str)
            if match:
                capacity = float(match.group(1))
                unit = match.group(2)
                return capacity * 1024 if unit == 'TB' else capacity
            return 0.0
        
        df_clean['Memory_Capacity_GB'] = df_clean['Memory'].apply(extract_memory_capacity)
        
        # Extract memory type
        def extract_memory_type(memory_str):
            if 'SSD' in memory_str:
                return 'SSD'
            elif 'HDD' in memory_str:
                return 'HDD'
            elif 'Flash' in memory_str:
                return 'Flash'
            elif 'Hybrid' in memory_str:
                return 'Hybrid'
            else:
                return 'Other'
        
        df_clean['Memory_Type'] = df_clean['Memory'].apply(extract_memory_type)
        
        print("✅ Memory features extracted")
        return df_clean
    
    def preprocess_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Fully preprocessed dataframe
        """
        print("🔧 Starting data preprocessing...")
        
        # Apply all preprocessing steps
        df_processed = self.clean_ram_weight(df)
        df_processed = self.extract_screen_features(df_processed)
        df_processed = self.extract_cpu_features(df_processed)
        df_processed = self.clean_memory(df_processed)
        
        # Drop original columns that have been processed
        columns_to_drop = ['ScreenResolution', 'Cpu', 'Memory']
        df_processed = df_processed.drop(columns=columns_to_drop)
        
        self.processed_data = df_processed
        print("✅ Data preprocessing completed")
        
        return df_processed
