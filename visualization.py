import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_distribution(df, target_col):
    """
    Plot target variable distribution with percentages
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
    """
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

def plot_correlation_heatmap(df, numerical_cols, threshold=0.7, top_n=5):
    """
    Plot correlation heatmap for numerical features
    
    Args:
        df (pd.DataFrame): Input dataframe
        numerical_cols (list): List of numerical column names
        threshold (float): Correlation threshold to display
        top_n (int): Number of top correlations to display
    """
    st.write("#### Correlation Heatmap of Strong Correlations Only")
    
    corr_matrix = df[numerical_cols].corr()
    
    # Filter strong correlations (but not self-correlation)
    strong_corr = corr_matrix.copy()
    strong_corr = strong_corr.where((corr_matrix.abs() >= threshold) & (corr_matrix.abs() != 1.0))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        strong_corr,
        annot=True,
        cmap="coolwarm",
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

def plot_boxplots(df, selected_num_cols):
    """
    Plot boxplots for selected numerical features
    
    Args:
        df (pd.DataFrame): Input dataframe
        selected_num_cols (list): List of selected numerical column names
    """
    st.write("#### Boxplot of Numerical Features")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df[selected_num_cols], palette='Set3', ax=ax)
    ax.set_title(f"Boxplots for Selected Numerical Features")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_categorical_features(df, selected_cat_cols):
    """
    Plot bar charts for selected categorical features
    
    Args:
        df (pd.DataFrame): Input dataframe
        selected_cat_cols (list): List of selected categorical column names
    """
    st.write("#### Bar Plot of Categorical Features")
    
    # Calculate the number of rows and columns for subplots
    n_cols = 2  # Number of plots per row
    n_rows = (len(selected_cat_cols) + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]  # Flatten in case of multiple rows
    
    for i, col in enumerate(selected_cat_cols):
        sns.countplot(data=df, x=col, ax=axes[i], palette='Set2')
        axes[i].set_title(f"Bar Plot for {col}")
        axes[i].set_ylabel("Count")
        axes[i].set_xlabel(col)
        axes[i].tick_params(axis='x', rotation=45)
    
    # Hide any unused subplots
    if len(selected_cat_cols) < len(axes):
        for j in range(len(selected_cat_cols), len(axes)):
            fig.delaxes(axes[j])
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_confusion_matrix(cm, labels):
    """
    Plot confusion matrix with percentages
    
    Args:
        cm (np.array): Confusion matrix
        labels (np.array): Class labels
    """
    # Calculate percentages for each cell in the confusion matrix
    cm_percentage = cm / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    
    # Add percentage text on top of the matrix
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j + 0.5, i + 0.5, f"{cm_percentage[i, j]:.1f}%", 
                    ha='center', va='center', color='black', fontsize=12)
    
    ax.set_title("Confusion Matrix", fontsize=16)
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    st.pyplot(fig)
