# Bias-mitigation-and-detection

# 🧠 Bias Detection and Mitigation Framework

This project is an interactive Streamlit-based web application designed to help users detect, visualize, and mitigate bias in machine learning datasets. It supports various fairness metrics and provides multiple mitigation strategies such as reweighting, resampling, fair representation learning, and adversarial debiasing using PyTorch.

---
---

## ⚙️ Workflow

1. **Upload CSV Dataset**  
   Upload your dataset (must include a binary target column and a sensitive attribute).

2. **Data Preprocessing**  
   - Missing values are handled.
   - Categorical features are label-encoded.

3. **Data Visualization**  
   - Target distribution, correlation heatmaps, boxplots, and category-wise bar plots.

4. **Model Training**  
   - Trains Logistic Regression, Random Forest, and SVM.
   - Best model is selected based on accuracy.

5. **Bias Detection Metrics**  
   - Statistical Parity Difference (SPD)  
   - Disparate Impact (DI)  
   - Equal Opportunity Difference (EOD)  
   - Average Odds Difference (AOD)

6. **Bias Mitigation Techniques**  
   - Reweighting  
   - Resampling using SMOTE and undersampling  
   - Fair Representation Learning (Autoencoder)  
   - Adversarial Debiasing (Neural Network with Gradient Reversal Layer)



8. **Download**  
   - Mitigated dataset can be downloaded for future use.
   - Download report for insights and results

---



You can install all required libraries with:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
git clone https://github.com/LunaticWrath07/Bias-mitigation-and-detection
cd bias-detection-framework
pip install -r requirements.txt
streamlit run main1.py
```
