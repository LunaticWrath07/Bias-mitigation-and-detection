import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import sqlite3
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# --- Custom Styling ---
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

    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Convert categorical variables to numeric using Label Encoding
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Step 3: Data Preview and Visualization
    st.subheader("Data Distribution")
    st.write("### Summary Statistics")
    st.write(df.describe())

    st.write("### Data Visualization After Preprocessing")
    
    # 1. Target Variable Distribution with Percentages
    st.write("### Target Variable Distribution")

    # Allow user to select Target Column
    target_col = st.selectbox("Select Target Column", df.columns)

    if target_col:
        st.write(f"#### Distribution of '{target_col}'")

        # Count occurrences and calculate percentages
        counts = df[target_col].value_counts()
        percentages = counts / counts.sum() * 100

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plotting the bar chart
        sns.barplot(x=counts.index, y=counts.values, palette="pastel", ax=ax)

        ax.set_title(f"Distribution of '{target_col}'", fontsize=16)
        ax.set_xlabel(target_col, fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        ax.tick_params(axis='x', rotation=45)

        # Display percentages on top of the bars
        for i, v in enumerate(counts.values):
            ax.text(i, v + 3, f"{percentages[i]:.1f}%", ha='center', fontsize=12)

        st.pyplot(fig)

        # Display counts and percentages as a table
        st.write("##### Counts and Percentages:")
        summary_df = pd.DataFrame({
            target_col: counts.index,
            "Count": counts.values,
            "Percentage": percentages.values
        })
        st.dataframe(summary_df)
    else:
        st.info("Please select a target column to visualize.")

    # 2. Correlation Heatmap + Top N Strongest Correlations
    st.write("#### Correlation Heatmap of Strong Correlations Only")

    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()

        # Set threshold and N
        threshold = st.slider("Set correlation threshold", 0.0, 1.0, 0.7)
        top_n = st.slider("How many top correlations to display?", 1, 20, 5)

        # Filter strong correlations (but not self-correlation)
        strong_corr = corr_matrix.copy()
        strong_corr = strong_corr.where((corr_matrix.abs() >= threshold) & (corr_matrix.abs() != 1.0))

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            strong_corr,
            annot=True,
            cmap=st.selectbox("Select Color Map", ["coolwarm"]),
            fmt=".2f",
            linewidths=0.5,
            linecolor='white',
            cbar=True,
            ax=ax,
            mask=strong_corr.isnull()
        )
        ax.set_title(f"Strong Correlations (>|{threshold}|)")
        st.pyplot(fig)

        # ---- TOP N strongest correlations table ----
        st.write("#### Top Strongest Correlations")

        # Extract upper triangle of correlation matrix without diagonal
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Flatten, drop nulls, take absolute value, sort
        sorted_corr = upper_triangle.stack().abs().sort_values(ascending=False)

        # Show top N
        if not sorted_corr.empty:
            top_corr_df = sorted_corr.head(top_n).reset_index()
            top_corr_df.columns = ['Feature 1', 'Feature 2', 'Correlation']
            st.dataframe(top_corr_df)
        else:
            st.info("No strong correlations found with the current threshold.")

    else:
        st.info("Need at least two numerical columns to plot a correlation heatmap.")

    # 3. Interactive Multi-Selection Boxplot for Numerical Features
    st.write("#### Boxplot of Numerical Features")

    if len(numerical_cols) > 0:
        selected_num_cols = st.multiselect("Select Numerical Columns to Plot", numerical_cols)

        if selected_num_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df[selected_num_cols], palette='Set3', ax=ax)
            ax.set_title(f"Boxplots for Selected Numerical Features")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("Please select at least one numerical column to display the boxplot.")
    else:
        st.info("No numerical columns available for boxplot.")

    # 4. Interactive Multi-Selection Bar Plot for Categorical Features
    st.write("#### Bar Plot of Categorical Features")

    # Filter categorical columns with <=10 unique values
    eligible_categorical_cols = [col for col in categorical_cols if df[col].nunique() <= 10]

    if eligible_categorical_cols:
        selected_cat_cols = st.multiselect("Select Categorical Columns to Plot", eligible_categorical_cols)

        if selected_cat_cols:
            # Calculate the number of rows and columns for subplots
            n_cols = 2  # Number of plots per row
            n_rows = (len(selected_cat_cols) + 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
            axes = axes.flatten()  # Flatten in case of multiple rows

            for i, col in enumerate(selected_cat_cols):
                sns.countplot(data=df, x=col, ax=axes[i], palette='Set2')
                axes[i].set_title(f"Bar Plot for {col}")
                axes[i].set_ylabel("Count")
                axes[i].set_xlabel(col)
                axes[i].tick_params(axis='x', rotation=45)

            # Hide any unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Please select at least one categorical column to display the plots.")
    else:
        st.info("No categorical columns with 10 or fewer unique values to plot.")

    # Step 4: Feature Selection (Auto-Selecting Features)
    st.subheader("Feature Selection")
    feature_cols = df.columns.tolist()
    feature_cols.remove(target_col)

    # Step 5: Model Training (Train Multiple Models and Select Best)
    st.subheader("Model Training and Selection")
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }

    best_model = None
    best_accuracy = 0

    X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=0.2, random_state=42)

    # Normalize Data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Store actual model objects and accuracy values
    model_objects = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store the model and its accuracy
        model_objects[model_name] = model

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    # Now `best_model` will hold the model object, not just a string
    st.write(f"### Best Model: {best_model} with Accuracy: {best_accuracy:.2f}")
    
    # After training and selecting the best model, calculate confusion matrix
    cm = confusion_matrix(y_test, best_model.predict(X_test))

    # Calculate percentages for each cell in the confusion matrix
    cm_percentage = cm / cm.sum(axis=1)[:, np.newaxis] * 100

    st.write("### Confusion Matrix")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), ax=ax)

    # Add percentage text on top of the matrix
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{cm_percentage[i, j]:.1f}%", ha='center', va='center', color='black', fontsize=12)

    ax.set_title("Confusion Matrix", fontsize=16)
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    st.pyplot(fig)

    # Step 6: Bias Detection Metrics and Mitigation
    st.subheader("Bias Detection Metrics")

    # Assuming sensitive_attr is the name of the sensitive attribute (e.g., gender, race)
    sensitive_attr = st.selectbox("Select Sensitive Attribute", df.columns)

    # Ensure that the sensitive attribute is present in the dataset
    if sensitive_attr in df.columns:
        # Convert X_train to DataFrame (retain original index)
        X_train_df = pd.DataFrame(X_train, index=df.index[:len(X_train)], columns=feature_cols)

        # Align df_train with X_train using the correct indices
        df_train = df.loc[X_train_df.index].copy()  
        df_train[sensitive_attr] = df.loc[X_train_df.index, sensitive_attr]  # Align the sensitive attribute
        df_train['target'] = y_train  # Add the target column

        # Identify Privileged & Unprivileged Groups
        privileged_value = df_train[sensitive_attr].mode()[0]  # Majority group
        privileged_group = df_train[df_train[sensitive_attr] == privileged_value]
        unprivileged_group = df_train[df_train[sensitive_attr] != privileged_value]

        # Prevent division errors
        if len(unprivileged_group) == 0 or len(privileged_group) == 0:
            st.warning("No instances found for either the privileged or unprivileged group.")
        else:
            # **Disparate Impact (DI)**
            di_unpriv = len(unprivileged_group[unprivileged_group['target'] == 1]) / len(unprivileged_group) if len(unprivileged_group) > 0 else 0
            di_priv = len(privileged_group[privileged_group['target'] == 1]) / len(privileged_group) if len(privileged_group) > 0 else 1
            disparate_impact = di_unpriv / di_priv if di_priv > 0 else 0

            # **Statistical Parity Difference (SPD)**
            spd = di_unpriv - di_priv

            # **Confusion Matrix (Ensure Correct Label Indexing)**
            y_pred = best_model.predict(X_test)  # Predict labels
            labels = np.unique(y_test)  # Get actual labels
            cm = confusion_matrix(y_test, y_pred, labels=labels)  

            # Check if the confusion matrix is correctly sized (at least 2x2)
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

                # **Display Fairness Metrics**
                st.write(f"**Disparate Impact:** {disparate_impact:.2f} (Ideal: 0.8 - 1.2)")
                st.write(f"**Statistical Parity Difference:** {spd:.2f} (Ideal: -0.1 to 0.1)")
                st.write(f"**Equal Opportunity Difference:** {eod:.2f} (Ideal: -0.1 to 0.1)")
                st.write(f"**Average Odds Difference:** {aod:.2f} (Ideal: -0.1 to 0.1)")
            else:
                st.warning("Confusion matrix dimensions are incorrect. Ensure binary classification.")

    def preprocess_data(df, target_col, sensitive_attr):
        """
        Ensure all features are numeric and handle categorical values dynamically.
        """
        # Convert categorical columns to numeric
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
        for col in categorical_cols:
            df[col] = encoders[col].transform(df[col])
        
        # Ensure no missing values
        df = df.dropna().reset_index(drop=True)
        
        return df, categorical_cols, encoders

    def apply_reweighting(df, target_col, sensitive_attr):
        """
        Applies reweighting by adjusting sample weights to balance both 
        the sensitive attribute and the target class distribution.
        Focuses on reducing Equal Opportunity Difference.
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
            smote = SMOTE(sampling_strategy="not majority", random_state=42)
            undersampler = RandomUnderSampler(sampling_strategy="majority", random_state=42)

            # Apply SMOTE first for oversampling minority groups within each sensitive group
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Apply undersampling within each sensitive group
            X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)
        else:
            X_resampled, y_resampled = X, y  # No resampling needed if balanced

        # Convert back to DataFrame
        df_resampled = pd.DataFrame(X_resampled, columns=feature_cols)
        df_resampled[target_col] = y_resampled

        # Restore the original sensitive attribute (if one-hot encoded)
        if sensitive_attr in categorical_cols:
            original_sensitive_values = df[[sensitive_attr]]
            df_resampled[sensitive_attr] = original_sensitive_values.iloc[:len(df_resampled)].values

        return df_resampled

    class FairAutoencoder(nn.Module):
        def __init__(self, input_dim, encoded_dim=8):
            super(FairAutoencoder, self).__init__()
            
            # Encoder: Compress input into a fair representation
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, encoded_dim)
            )
            
            # Decoder: Reconstruct input from fair representation
            self.decoder = nn.Sequential(
                nn.Linear(encoded_dim, 32),
                nn.ReLU(),
                nn.Linear(32, input_dim)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded

    def fairness_penalty(encoded, sensitive):
        """
        Fairness loss function: Enforces independence between encoded representation and sensitive attribute.
        Uses Maximum Mean Discrepancy (MMD) to measure similarity between distributions.
        """
        sensitive_0 = encoded[sensitive == 0]
        sensitive_1 = encoded[sensitive == 1]

        mean_0 = sensitive_0.mean(dim=0) if len(sensitive_0) > 0 else torch.zeros(encoded.shape[1])
        mean_1 = sensitive_1.mean(dim=0) if len(sensitive_1) > 0 else torch.zeros(encoded.shape[1])

        return torch.norm(mean_0 - mean_1, p=2)  # L2 Distance between means

    def apply_fair_representation(df, target_col, sensitive_attr, encoded_dim=8, epochs=100, lr=0.001, lambda_fair=0.1):
        """
        Learns a fair representation of the dataset using a modified autoencoder with fairness constraints.
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

    class GradientReversalFunction(torch.autograd.Function):
        """
        Implements the Gradient Reversal Layer (GRL).
        This forces the feature extractor to minimize task loss while maximizing adversary loss.
        """
        @staticmethod
        def forward(ctx, x, lambda_grl):
            ctx.lambda_grl = lambda_grl
            return x

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.lambda_grl * grad_output, None  # Reverses gradient

    class FeatureExtractor(nn.Module):
        def __init__(self, input_dim, encoded_dim=8):
            super(FeatureExtractor, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, encoded_dim)
            )

        def forward(self, x, lambda_grl=1.0):
            x = self.network(x)
            return GradientReversalFunction.apply(x, lambda_grl)

    class Adversary(nn.Module):
        def __init__(self, input_dim):
            super(Adversary, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.network(x)

    def apply_adversarial_debiasing(df, target_col, sensitive_attr, encoded_dim=8, epochs=100, lr_extractor=0.001, lr_adversary=0.005, lambda_grl=1.0):
        """
        Uses adversarial debiasing with a Gradient Reversal Layer to remove sensitive attribute influence.
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
        """
        df, categorical_cols, encoders = preprocess_data(df, target_col, sensitive_attr)

        if mitigation_choice == "Reweighting":
            df_transformed = apply_reweighting(df, target_col, sensitive_attr)
        elif mitigation_choice == "Resampling":
            df_transformed = apply_resampling(df, target_col, sensitive_attr)
        elif mitigation_choice == "Fair Representation Learning":
            df_transformed = apply_fair_representation(df, target_col, sensitive_attr)
        elif mitigation_choice == "Adversarial Debiasing":
            df_transformed = apply_adversarial_debiasing(df, target_col, sensitive_attr)
        else:
            df_transformed = df.copy()

        return df_transformed

    # Step 7: Bias Mitigation (Reweighting and Resampling)
    st.subheader("Bias Mitigation Techniques")
    mitigation_choice = st.selectbox("Choose Mitigation Method", ["Reweighting", "Resampling", "Fair Representation Learning", "Adversarial Debiasing"])
    
    # Apply the chosen mitigation method
    mitigated_data = apply_mitigation(df, target_col, sensitive_attr, mitigation_choice)

    # Show the mitigated data (after applying resampling or reweighting)
    st.write("### Mitigated Data Preview")
    st.write(mitigated_data.head())
    
    if mitigation_choice:
        st.subheader("Fairness Metrics After Mitigation")

        # Apply mitigation technique
        mitigated_data = apply_mitigation(df, target_col, sensitive_attr, mitigation_choice)
        
        # Check if sensitive attribute exists after mitigation
        if sensitive_attr not in mitigated_data.columns:
            st.error(f"Sensitive attribute '{sensitive_attr}' not found after mitigation!")
            st.stop()

        # Get feature columns dynamically
        feature_cols = [col for col in mitigated_data.columns if col not in [target_col, sensitive_attr]]

        if not feature_cols:
            st.error("No valid feature columns found after mitigation!")
        else:
            # Split the mitigated dataset
            X_train, X_test, y_train, y_test = train_test_split(
                mitigated_data[feature_cols], mitigated_data[target_col], test_size=0.2, random_state=42
            )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train model on mitigated data
            best_model.fit(X_train, y_train)

            # Make predictions
            y_pred = best_model.predict(X_test)

            # Compute confusion matrix
            cm_mitigated = confusion_matrix(y_test, y_pred)

            # Fairness Metrics Calculation
            if len(mitigated_data[sensitive_attr].unique()) == 2:
                unprivileged_group = mitigated_data[mitigated_data[sensitive_attr] == 0]
                privileged_group = mitigated_data[mitigated_data[sensitive_attr] == 1]

                di_unpriv_mitigated = len(unprivileged_group[unprivileged_group[target_col] == 1]) / len(unprivileged_group) if len(unprivileged_group) > 0 else 0
                di_priv_mitigated = len(privileged_group[privileged_group[target_col] == 1]) / len(privileged_group) if len(privileged_group) > 0 else np.nan

                # Prevent zero-division error
                disparate_impact_mitigated = np.nan_to_num(di_unpriv_mitigated / di_priv_mitigated, nan=0.0)
                spd_mitigated = di_unpriv_mitigated - di_priv_mitigated
            else:
                st.warning("Sensitive attribute is not binary. Cannot calculate DI and SPD.")
                disparate_impact_mitigated = 0
                spd_mitigated = 0

            # Equal Opportunity Difference
            if cm_mitigated.shape == (2, 2):  
                tp_priv_mitigated = cm_mitigated[1, 1]
                fn_priv_mitigated = cm_mitigated[1, 0]
                tp_unpriv_mitigated = cm_mitigated[0, 1]
                fn_unpriv_mitigated = cm_mitigated[0, 0]
            else:
                tp_priv_mitigated = fn_priv_mitigated = tp_unpriv_mitigated = fn_unpriv_mitigated = 0

            tpr_priv_mitigated = tp_priv_mitigated / (tp_priv_mitigated + fn_priv_mitigated) if (tp_priv_mitigated + fn_priv_mitigated) > 0 else 0
            tpr_unpriv_mitigated = tp_unpriv_mitigated / (tp_unpriv_mitigated + fn_unpriv_mitigated) if (tp_unpriv_mitigated + fn_unpriv_mitigated) > 0 else 0
            eod_mitigated = tpr_unpriv_mitigated - tpr_priv_mitigated

            # Average Odds Difference
            fp_priv_mitigated = cm_mitigated[1, 0] if len(cm_mitigated) > 1 else 0
            fp_unpriv_mitigated = cm_mitigated[0, 0] if len(cm_mitigated) > 1 else 0
            tn_priv_mitigated = cm_mitigated[1, 1] if len(cm_mitigated) > 1 else 0
            tn_unpriv_mitigated = cm_mitigated[0, 1] if len(cm_mitigated) > 1 else 0

            fpr_priv_mitigated = fp_priv_mitigated / (fp_priv_mitigated + tn_priv_mitigated) if (fp_priv_mitigated + tn_priv_mitigated) > 0 else 0
            fpr_unpriv_mitigated = fp_unpriv_mitigated / (fp_unpriv_mitigated + tn_unpriv_mitigated) if (fp_unpriv_mitigated + tn_unpriv_mitigated) > 0 else 0

            aod_mitigated = ((tpr_unpriv_mitigated - tpr_priv_mitigated) + (fpr_unpriv_mitigated - fpr_priv_mitigated)) / 2

            # Display Metrics
            st.write(f"**Disparate Impact After Mitigation:** {disparate_impact_mitigated:.2f} (Ideal: 0.8 - 1.2)")
            st.write(f"**Statistical Parity Difference After Mitigation:** {spd_mitigated:.2f} (Ideal: -0.1 to 0.1)")
            st.write(f"**Equal Opportunity Difference After Mitigation:** {eod_mitigated:.2f} (Ideal: -0.1 to 0.1)")
            st.write(f"**Average Odds Difference After Mitigation:** {aod_mitigated:.2f} (Ideal: -0.1 to 0.1)")
            
            # **Post-Mitigation Visualization: Confusion Matrix**
            st.write("### Confusion Matrix After Mitigation")
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(cm_mitigated, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), ax=ax)
            ax.set_title("Confusion Matrix After Mitigation")
            st.pyplot(fig)

            # **Post-Mitigation Visualization: Target Variable Distribution**
            st.write("### Distribution of Target Variable After Mitigation")
            fig, ax = plt.subplots(figsize=(8, 6))
            mitigated_data[target_col].value_counts().plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title("Target Variable Distribution After Mitigation")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    # Step 8: Data Preview after Mitigation
    st.write("### Data Preview after Mitigation")
    st.write(mitigated_data.head())

    # Step 9: Download Mitigated Dataset
    st.subheader("Download Mitigated Dataset")
    csv = mitigated_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Mitigated Data", data=csv, file_name="mitigated_dataset.csv", mime="text/csv")

    # Step 10: Download Report
    st.subheader("Download Report")
    
    # Function to create PDF report
    def create_pdf_report(bias_metrics, mitigation_method, fairness_metrics):
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Title
        p.setFont("Helvetica-Bold", 16)
        p.drawString(100, height - 50, "Bias Detection and Mitigation Report")

        # Bias Detection Metrics
        p.setFont("Helvetica-Bold", 14)
        p.drawString(100, height - 100, "Bias Detection Metrics:")
        p.setFont("Helvetica", 12)
        for metric, value in bias_metrics.items():
            p.drawString(100, height - 120 - 20 * list(bias_metrics.keys()).index(metric), f"{metric}: {value:.2f}")

        

        # Mitigation Method
        p.setFont("Helvetica-Bold", 14)
        p.drawString(100, height - 260, "Mitigation Method Selected:")
        p.setFont("Helvetica", 12)
        p.drawString(100, height - 280, mitigation_method)

        # Fairness Metrics After Mitigation
        p.setFont("Helvetica-Bold", 14)
        p.drawString(100, height - 320, "Fairness Metrics After Mitigation:")
        p.setFont("Helvetica", 12)
        for metric, value in fairness_metrics.items():
            p.drawString(100, height - 340 - 20 * list(fairness_metrics.keys()).index(metric), f"{metric}: {value:.2f}")

        p.showPage()
        p.save()
        buffer.seek(0)
        return buffer

    # Prepare data for the report
    bias_metrics = {
        "Disparate Impact": disparate_impact,
        "Statistical Parity Difference": spd,
        "Equal Opportunity Difference": eod,
        "Average Odds Difference": aod
    }
    
    
    fairness_metrics = {
        "Disparate Impact After Mitigation": disparate_impact_mitigated,
        "Statistical Parity Difference After Mitigation": spd_mitigated,
        "Equal Opportunity Difference After Mitigation": eod_mitigated,
        "Average Odds Difference After Mitigation": aod_mitigated
    }

    # Add a button to download the report
    if st.button("Download Report"):
        # Create PDF report
        pdf_report = create_pdf_report(bias_metrics, mitigation_choice, fairness_metrics)

        # Download the PDF
        st.download_button("Download PDF Report", pdf_report, "report.pdf", "application/pdf")
