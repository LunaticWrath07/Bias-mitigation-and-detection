import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """
    Perform data preprocessing:
    - Identify categorical and numerical columns
    - Convert categorical variables to numeric
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (processed_df, categorical_cols, numerical_cols)
    """
    df = df.copy()
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Convert categorical variables to numeric using Label Encoding
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    return df, categorical_cols, numerical_cols

def preprocess_data_with_sensitive(df, target_col, sensitive_attr):
    """
    Ensure all features are numeric and handle categorical values dynamically.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        sensitive_attr (str): Sensitive attribute column name
        
    Returns:
        tuple: (processed_df, categorical_cols, encoders)
    """
    df = df.copy()
    
    # Convert categorical columns to numeric
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
    
    for col in categorical_cols:
        df[col] = encoders[col].transform(df[col])
    
    # Ensure no missing values
    df = df.dropna().reset_index(drop=True)
    
    return df, categorical_cols, encoders
