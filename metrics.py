import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_bias_metrics(df, X_train, y_train, sensitive_attr, target_col, model, X_test, y_test):
    """
    Calculate bias detection metrics
    
    Args:
        df (pd.DataFrame): Input dataframe
        X_train: Training features
        y_train: Training target values
        sensitive_attr (str): Sensitive attribute column name
        target_col (str): Target column name
        model: Trained model
        X_test: Test features
        y_test: Test target values
        
    Returns:
        dict: Dictionary containing bias metrics
    """
    try:
        # Convert X_train to DataFrame (retain original index)
        X_train_df = pd.DataFrame(X_train, index=df.index[:len(X_train)])
        
        # Align df_train with X_train using the correct indices
        df_train = df.loc[X_train_df.index].copy()
        df_train[sensitive_attr] = df.loc[X_train_df.index, sensitive_attr]  # Align sensitive attribute
        df_train['target'] = y_train  # Add target column
        
        # Identify Privileged & Unprivileged Groups
        privileged_value = df_train[sensitive_attr].mode()[0]  # Majority group
        privileged_group = df_train[df_train[sensitive_attr] == privileged_value]
        unprivileged_group = df_train[df_train[sensitive_attr] != privileged_value]
        
        # Prevent division errors
        if len(unprivileged_group) == 0 or len(privileged_group) == 0:
            return None
        
        # **Disparate Impact (DI)**
        di_unpriv = len(unprivileged_group[unprivileged_group['target'] == 1]) / len(unprivileged_group) if len(unprivileged_group) > 0 else 0
        di_priv = len(privileged_group[privileged_group['target'] == 1]) / len(privileged_group) if len(privileged_group) > 0 else 1
        disparate_impact = di_unpriv / di_priv if di_priv > 0 else 0
        
        # **Statistical Parity Difference (SPD)**
        spd = di_unpriv - di_priv
        
        # **Confusion Matrix (Ensure Correct Label Indexing)**
        y_pred = model.predict(X_test)  # Predict labels
        labels = np.unique(y_test)  # Get actual labels
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        
        # Check if the confusion matrix is correctly sized
        if cm.shape[0] >= 2:
            tp_priv = cm[1, 1]  # True Positives for privileged
            fn_priv = cm[1, 0]  # False Negatives for privileged
            tp_unpriv = cm[0, 1]  # True Positives for unprivileged
            fn_unpriv = cm[0, 0]  # False Negatives for unprivileged
            
            # **Equal Opportunity Difference (EOD)**
            tpr_priv = tp_priv / (tp_priv + fn_priv) if (tp_priv + fn_priv) > 0 else 0
            tpr_unpriv = tp_unpriv / (tp_unpriv + fn_unpriv) if (tp_unpriv + fn_unpriv) > 0 else 0
            eod = tpr_unpriv - tpr_priv
            
            # **False Positive Rate Calculation for Average Odds Difference (AOD)**
            fp_priv = cm[1, 0]  # False Positives for privileged
            fp_unpriv = cm[0, 0]  # False Positives for unprivileged
            
            fpr_priv = fp_priv / (fp_priv + tp_priv) if (fp_priv + tp_priv) > 0 else 0
            fpr_unpriv = fp_unpriv / (fp_unpriv + tp_unpriv) if (fp_unpriv + tp_unpriv) > 0 else 0
            
            # **Average Odds Difference (AOD)**
            aod = ((tpr_unpriv - tpr_priv) + (fpr_unpriv - fpr_priv)) / 2
            
            return {
                'disparate_impact': disparate_impact,
                'statistical_parity_difference': spd,
                'equal_opportunity_difference': eod,
                'average_odds_difference': aod
            }
        
        return None
    
    except Exception as e:
        print(f"Error calculating bias metrics: {e}")
        return None
