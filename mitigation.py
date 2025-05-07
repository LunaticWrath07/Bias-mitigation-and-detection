import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Import classes from models.py
from models import FairAutoencoder, FeatureExtractor, Adversary, fairness_penalty

def preprocess_data(df, target_col=None, sensitive_attr=None):
    """
    Ensure all features are numeric and handle categorical values dynamically.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    target_col : str, optional
        The name of the target column
    sensitive_attr : str, optional
        The name of the sensitive attribute column
        
    Returns:
    --------
    tuple
        (processed_df, categorical_cols, encoders)
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Convert categorical columns to numeric
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        encoders[col] = le
    
    # Ensure no missing values
    df_processed = df_processed.dropna().reset_index(drop=True)
    
    return df_processed, categorical_cols, encoders

def apply_reweighting(df, target_col, sensitive_attr):
    """
    Applies reweighting by adjusting sample weights to balance both 
    the sensitive attribute and the target class distribution.
    Focuses on reducing Equal Opportunity Difference.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    target_col : str
        The name of the target column
    sensitive_attr : str
        The name of the sensitive attribute column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added sample_weight column
    """
    df = df.copy()

    # Compute joint probability of (Sensitive Attribute, Target)
    joint_dist = df.groupby([sensitive_attr, target_col]).size() / len(df)

    # Compute overall probability of sensitive attribute & target class
    sensitive_dist = df[sensitive_attr].value_counts(normalize=True).to_dict()
    target_dist = df[target_col].value_counts(normalize=True).to_dict()

    # Calculate positive outcome probabilities for each sensitive group
    positive_outcomes = df[df[target_col] == 1].groupby(sensitive_attr).size()
    positive_prob = (positive_outcomes / positive_outcomes.sum()).to_dict()

    # Compute weights: Normalize by both joint probability & positive outcome fairness
    df["sample_weight"] = df.apply(
        lambda row: ((1 / joint_dist.get((row[sensitive_attr], row[target_col]), 1e-6)) /
                    ((1 / sensitive_dist[row[sensitive_attr]]) * (1 / target_dist[row[target_col]]))) *
                    (1 / positive_prob.get(row[sensitive_attr], 1e-6) if row[target_col] == 1 else 1),
        axis=1
    )

    # Normalize sample weights to sum to dataset size
    df["sample_weight"] /= df["sample_weight"].sum() / len(df)

    return df

def apply_resampling(df, target_col, sensitive_attr):
    """
    Balances the dataset using a combination of oversampling (for underrepresented groups)
    and undersampling (for overrepresented groups) while preserving fairness.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    target_col : str
        The name of the target column
    sensitive_attr : str
        The name of the sensitive attribute column
        
    Returns:
    --------
    pandas.DataFrame
        Resampled dataframe with balanced classes
    """
    df = df.copy()

    # Identify underrepresented and overrepresented groups
    group_counts = df.groupby([sensitive_attr, target_col]).size().unstack(fill_value=0)
    
    # Find the minority class size across all groups
    min_count = group_counts.min().min()
    max_count = group_counts.max().max()

    # Encode categorical sensitive attributes (SMOTE requires numerical values)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=[sensitive_attr] if sensitive_attr in categorical_cols else [])

    # Separate features and target
    feature_cols = [col for col in df_encoded.columns if col not in [target_col]]
    X, y = df_encoded[feature_cols], df[target_col]

    if min_count < max_count:  # Apply resampling only if imbalance exists
        # Apply SMOTE first for oversampling minority groups
        smote = SMOTE(sampling_strategy="not majority", random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Apply undersampling within each sensitive group
        undersampler = RandomUnderSampler(sampling_strategy="majority", random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)
    else:
        X_resampled, y_resampled = X, y  # No resampling needed if balanced

    # Convert back to DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=feature_cols)
    df_resampled[target_col] = y_resampled

    # Restore the original sensitive attribute (if one-hot encoded)
    if sensitive_attr in categorical_cols:
        # If we have a smaller or equal sized dataset after resampling, take the first n values
        if len(df_resampled) <= len(df):
            df_resampled[sensitive_attr] = df[sensitive_attr].values[:len(df_resampled)]
        else:
            # If we have a larger dataset, we need to repeat values to match the new size
            # This is a simplification and might not preserve the exact distribution
            df_resampled[sensitive_attr] = np.concatenate([
                df[sensitive_attr].values,
                df[sensitive_attr].sample(len(df_resampled) - len(df), replace=True).values
            ])

    return df_resampled

def apply_fair_representation(df, target_col, sensitive_attr, encoded_dim=8, epochs=100, lr=0.001, lambda_fair=0.1):
    """
    Learns a fair representation of the dataset using a modified autoencoder with fairness constraints.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    target_col : str
        The name of the target column
    sensitive_attr : str
        The name of the sensitive attribute column
    encoded_dim : int, optional
        Dimension of the encoded representation
    epochs : int, optional
        Number of training epochs
    lr : float, optional
        Learning rate
    lambda_fair : float, optional
        Weight for fairness penalty
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with fair representations
    """
    df = df.copy()
    feature_cols = [col for col in df.columns if col not in [target_col, sensitive_attr]]

    # Convert Data to Torch Tensors
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    sensitive = torch.tensor(df[sensitive_attr].values, dtype=torch.float32)

    # Initialize Fair Autoencoder
    autoencoder = FairAutoencoder(input_dim=X.shape[1], encoded_dim=encoded_dim)
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        encoded, decoded = autoencoder(X)
        loss_reconstruction = criterion(decoded, X)
        loss_fairness = fairness_penalty(encoded, sensitive)

        # Total Loss = Reconstruction Loss + Fairness Regularization
        loss = loss_reconstruction + lambda_fair * loss_fairness

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Fairness Penalty: {loss_fairness.item():.4f}")

    # Extract Fair Representations
    with torch.no_grad():
        encoded_features = autoencoder.encoder(X).numpy()
        encoded_feature_names = [f"encoded_{i}" for i in range(encoded_features.shape[1])]
        df_transformed = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)

    # Add back Target & Sensitive Attributes
    df_transformed = pd.concat([df[[target_col, sensitive_attr]], df_transformed], axis=1)

    return df_transformed

def apply_adversarial_debiasing(df, target_col, sensitive_attr, encoded_dim=8, epochs=100, lr_extractor=0.001, lr_adversary=0.005, lambda_grl=1.0):
    """
    Uses adversarial debiasing with a Gradient Reversal Layer to remove sensitive attribute influence.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    target_col : str
        The name of the target column
    sensitive_attr : str
        The name of the sensitive attribute column
    encoded_dim : int, optional
        Dimension of the encoded representation
    epochs : int, optional
        Number of training epochs
    lr_extractor : float, optional
        Learning rate for feature extractor
    lr_adversary : float, optional
        Learning rate for adversary
    lambda_grl : float, optional
        Weight for gradient reversal
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with debiased features
    """
    df = df.copy()
    feature_cols = [col for col in df.columns if col not in [target_col, sensitive_attr]]

    # Convert Data to Torch Tensors
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    S = torch.tensor(df[sensitive_attr].values, dtype=torch.float32).view(-1, 1)

    # Initialize Feature Extractor & Adversary
    extractor = FeatureExtractor(X.shape[1], encoded_dim=encoded_dim)
    adversary = Adversary(encoded_dim)

    optimizer_extractor = optim.Adam(extractor.parameters(), lr=lr_extractor)
    optimizer_adversary = optim.Adam(adversary.parameters(), lr=lr_adversary)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        # Train feature extractor with Gradient Reversal
        optimizer_extractor.zero_grad()
        features = extractor(X, lambda_grl=lambda_grl)
        adversary_preds = adversary(features)
        loss_extractor = -criterion(adversary_preds, S)  # Maximize adversary loss (via GRL)
        loss_extractor.backward()
        optimizer_extractor.step()

        # Train adversary
        optimizer_adversary.zero_grad()
        adversary_preds = adversary(extractor(X, lambda_grl=0).detach())  # No gradient reversal for adversary
        loss_adversary = criterion(adversary_preds, S)
        loss_adversary.backward()
        optimizer_adversary.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Extractor Loss: {loss_extractor.item():.4f}, Adversary Loss: {loss_adversary.item():.4f}")

    # Extract Debiased Features
    with torch.no_grad():
        transformed_features = extractor(X, lambda_grl=0).numpy()
        transformed_feature_names = [f"transformed_{i}" for i in range(transformed_features.shape[1])]
        df_transformed = pd.DataFrame(transformed_features, columns=transformed_feature_names, index=df.index)

    # Add back Target & Sensitive Attributes
    df_transformed = pd.concat([df[[target_col, sensitive_attr]], df_transformed], axis=1)

    return df_transformed

def apply_mitigation(df, target_col, sensitive_attr, mitigation_choice):
    """
    Apply the selected bias mitigation technique.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    target_col : str
        The name of the target column
    sensitive_attr : str
        The name of the sensitive attribute column
    mitigation_choice : str
        The name of the mitigation technique
        
    Returns:
    --------
    pandas.DataFrame
        Mitigated dataframe
    """
    df_processed, categorical_cols, encoders = preprocess_data(df, target_col, sensitive_attr)

    if mitigation_choice == "Reweighting":
        df_transformed = apply_reweighting(df_processed, target_col, sensitive_attr)
    elif mitigation_choice == "Resampling":
        df_transformed = apply_resampling(df_processed, target_col, sensitive_attr)
    elif mitigation_choice == "Fair Representation Learning":
        df_transformed = apply_fair_representation(df_processed, target_col, sensitive_attr)
    elif mitigation_choice == "Adversarial Debiasing":
        df_transformed = apply_adversarial_debiasing(df_processed, target_col, sensitive_attr)
    else:
        df_transformed = df_processed.copy()

    return df_transformed
