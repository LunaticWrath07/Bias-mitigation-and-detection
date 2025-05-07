import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Import custom modules
from preprocessing import preprocess_data
from visualization import (
    plot_target_distribution, 
    plot_correlation_heatmap, 
    plot_boxplots, 
    plot_categorical_features,
    plot_confusion_matrix
)
from metrics import calculate_bias_metrics
from mitigation import apply_mitigation
from utils import create_pdf_report

# --- Page Configuration ---
st.set_page_config(page_title="Bias Detection & Mitigation", layout="wide")

# --- Header with Logo ---
col1, col2 = st.columns([1, 4])
with col1:
    st.image("logo.png", width=150)  # Replace with your logo
with col2:
    st.title("Bias Detection and Mitigation")

# Step 1: Dataset Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())

    # Step 2: Preprocessing
    st.subheader("Data Preprocessing")
    
    # Handle missing values
    df.fillna(df.mode().iloc[0], inplace=True)
    
    # Convert categorical variables to numeric
    df, categorical_cols, numerical_cols = preprocess_data(df)
    
    # Step 3: Data Preview and Visualization
    st.subheader("Data Distribution")
    st.write("### Summary Statistics")
    st.write(df.describe())

    st.write("### Data Visualization After Preprocessing")
    
    # Allow user to select Target Column
    target_col = st.selectbox("Select Target Column", df.columns)

    if target_col:
        # Plot target distribution
        plot_target_distribution(df, target_col)
        
        # Correlation heatmap
        if len(numerical_cols) > 1:
            threshold = st.slider("Set correlation threshold", 0.0, 1.0, 0.7)
            top_n = st.slider("How many top correlations to display?", 1, 20, 5)
            plot_correlation_heatmap(df, numerical_cols, threshold, top_n)
        else:
            st.info("Need at least two numerical columns to plot a correlation heatmap.")
        
        # Boxplots for numerical features
        if len(numerical_cols) > 0:
            selected_num_cols = st.multiselect("Select Numerical Columns to Plot", numerical_cols)
            if selected_num_cols:
                plot_boxplots(df, selected_num_cols)
            else:
                st.info("Please select at least one numerical column to display the boxplot.")
        else:
            st.info("No numerical columns available for boxplot.")
        
        # Bar plots for categorical features
        eligible_categorical_cols = [col for col in categorical_cols if df[col].nunique() <= 10]
        if eligible_categorical_cols:
            selected_cat_cols = st.multiselect("Select Categorical Columns to Plot", eligible_categorical_cols)
            if selected_cat_cols:
                plot_categorical_features(df, selected_cat_cols)
            else:
                st.info("Please select at least one categorical column to display the plots.")
        else:
            st.info("No categorical columns with 10 or fewer unique values to plot.")
        
        # Step 4: Feature Selection
        st.subheader("Feature Selection")
        feature_cols = df.columns.tolist()
        feature_cols.remove(target_col)
        
        # Step 5: Model Training
        st.subheader("Model Training and Selection")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df[feature_cols], df[target_col], test_size=0.2, random_state=42
        )
        
        # Normalize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models (import models from models.py)
        from models import train_and_select_model
        
        best_model, best_accuracy = train_and_select_model(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Show results
        st.write(f"### Best Model: {type(best_model).__name__} with Accuracy: {best_accuracy:.2f}")
        
        # Calculate and plot confusion matrix
        y_pred = best_model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        
        st.write("### Confusion Matrix")
        plot_confusion_matrix(cm, np.unique(y_test))
        
        # Step 6: Bias Detection Metrics
        st.subheader("Bias Detection Metrics")
        
        # Select sensitive attribute
        sensitive_attr = st.selectbox("Select Sensitive Attribute", df.columns)
        
        # Calculate bias metrics
        if sensitive_attr in df.columns:
            bias_metrics = calculate_bias_metrics(df, X_train, y_train, sensitive_attr, target_col, best_model, X_test, y_test)
            
            if bias_metrics:
                disparate_impact = bias_metrics.get('disparate_impact', 0)
                spd = bias_metrics.get('statistical_parity_difference', 0)
                eod = bias_metrics.get('equal_opportunity_difference', 0)
                aod = bias_metrics.get('average_odds_difference', 0)
                
                # Display fairness metrics
                st.write(f"**Disparate Impact:** {disparate_impact:.2f} (Ideal: 0.8 - 1.2)")
                st.write(f"**Statistical Parity Difference:** {spd:.2f} (Ideal: -0.1 to 0.1)")
                st.write(f"**Equal Opportunity Difference:** {eod:.2f} (Ideal: -0.1 to 0.1)")
                st.write(f"**Average Odds Difference:** {aod:.2f} (Ideal: -0.1 to 0.1)")
            else:
                st.warning("Could not calculate bias metrics. Ensure binary classification and proper data structure.")
        
        # Step 7: Bias Mitigation
        st.subheader("Bias Mitigation Techniques")
        mitigation_choice = st.selectbox(
            "Choose Mitigation Method", 
            ["Reweighting", "Resampling", "Fair Representation Learning", "Adversarial Debiasing"]
        )
        
        # Apply mitigation
        mitigated_data = apply_mitigation(df, target_col, sensitive_attr, mitigation_choice)
        
        # Show mitigated data preview
        st.write("### Mitigated Data Preview")
        st.write(mitigated_data.head())
        
        # Calculate fairness metrics after mitigation
        if mitigation_choice and sensitive_attr in mitigated_data.columns:
            st.subheader("Fairness Metrics After Mitigation")
            
            # Get feature columns dynamically
            feature_cols = [col for col in mitigated_data.columns if col not in [target_col, sensitive_attr]]
            
            if feature_cols:
                # Split mitigated dataset
                X_train_mit, X_test_mit, y_train_mit, y_test_mit = train_test_split(
                    mitigated_data[feature_cols], mitigated_data[target_col], test_size=0.2, random_state=42
                )
                
                # Scale features
                scaler_mit = StandardScaler()
                X_train_mit_scaled = scaler_mit.fit_transform(X_train_mit)
                X_test_mit_scaled = scaler_mit.transform(X_test_mit)
                
                # Train model on mitigated data
                best_model.fit(X_train_mit_scaled, y_train_mit)
                
                # Make predictions
                y_pred_mit = best_model.predict(X_test_mit_scaled)
                
                # Compute confusion matrix
                cm_mitigated = confusion_matrix(y_test_mit, y_pred_mit)
                
                # Calculate post-mitigation metrics
                metrics_after = calculate_bias_metrics(
                    mitigated_data, X_train_mit, y_train_mit, 
                    sensitive_attr, target_col, best_model, 
                    X_test_mit_scaled, y_test_mit
                )
                
                if metrics_after:
                    disparate_impact_mitigated = metrics_after.get('disparate_impact', 0)
                    spd_mitigated = metrics_after.get('statistical_parity_difference', 0)
                    eod_mitigated = metrics_after.get('equal_opportunity_difference', 0)
                    aod_mitigated = metrics_after.get('average_odds_difference', 0)
                    
                    # Display metrics
                    st.write(f"**Disparate Impact After Mitigation:** {disparate_impact_mitigated:.2f} (Ideal: 0.8 - 1.2)")
                    st.write(f"**Statistical Parity Difference After Mitigation:** {spd_mitigated:.2f} (Ideal: -0.1 to 0.1)")
                    st.write(f"**Equal Opportunity Difference After Mitigation:** {eod_mitigated:.2f} (Ideal: -0.1 to 0.1)")
                    st.write(f"**Average Odds Difference After Mitigation:** {aod_mitigated:.2f} (Ideal: -0.1 to 0.1)")
                    
                    # Visualize post-mitigation confusion matrix
                    st.write("### Confusion Matrix After Mitigation")
                    plot_confusion_matrix(cm_mitigated, np.unique(y_test_mit))
                    
                    # Target variable distribution after mitigation
                    st.write("### Distribution of Target Variable After Mitigation")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    mitigated_data[target_col].value_counts().plot(kind='bar', ax=ax, color='skyblue')
                    ax.set_title("Target Variable Distribution After Mitigation")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
                    
                    # Step 8: Download options
                    st.subheader("Download Mitigated Dataset")
                    csv = mitigated_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Mitigated Data", 
                        data=csv, 
                        file_name="mitigated_dataset.csv", 
                        mime="text/csv"
                    )
                    
                    # Step 9: PDF Report
                    st.subheader("Download Report")
                    
                    # Prepare data for the report
                    bias_metrics_dict = {
                        "Disparate Impact": disparate_impact,
                        "Statistical Parity Difference": spd,
                        "Equal Opportunity Difference": eod,
                        "Average Odds Difference": aod
                    }
                    
                    fairness_metrics_dict = {
                        "Disparate Impact After Mitigation": disparate_impact_mitigated,
                        "Statistical Parity Difference After Mitigation": spd_mitigated,
                        "Equal Opportunity Difference After Mitigation": eod_mitigated,
                        "Average Odds Difference After Mitigation": aod_mitigated
                    }
                    
                    # Add a button to download the report
                    if st.button("Download Report"):
                        # Create PDF report
                        pdf_report = create_pdf_report(bias_metrics_dict, mitigation_choice, fairness_metrics_dict)
                        
                        # Download the PDF
                        st.download_button(
                            "Download PDF Report", 
                            pdf_report, 
                            "report.pdf", 
                            "application/pdf"
                        )
            else:
                st.error("No valid feature columns found after mitigation!")
        else:
            st.error(f"Sensitive attribute '{sensitive_attr}' not found after mitigation!")
