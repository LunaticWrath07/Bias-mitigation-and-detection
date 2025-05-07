import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

def train_and_select_model(X_train, y_train, X_test, y_test):
    """
    Train multiple models and select the best performing one based on accuracy.
    
    Parameters:
    -----------
    X_train : array-like
        Training feature data
    y_train : array-like
        Training target data
    X_test : array-like
        Test feature data
    y_test : array-like
        Test target data
        
    Returns:
    --------
    tuple
        (best_model, best_accuracy)
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }
    
    best_model = None
    best_accuracy = 0
    
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Update best model if this one is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    return best_model, best_accuracy

class FairAutoencoder(nn.Module):
    """
    Autoencoder for learning fair representations of data.
    """
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
    """
    Neural network for extracting features with gradient reversal capability.
    """
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
    """
    Adversary network for adversarial debiasing.
    """
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

def fairness_penalty(encoded, sensitive):
    """
    Fairness loss function: Enforces independence between encoded representation and sensitive attribute.
    Uses Maximum Mean Discrepancy (MMD) to measure similarity between distributions.
    
    Parameters:
    -----------
    encoded : torch.Tensor
        Encoded features
    sensitive : torch.Tensor
        Sensitive attribute values
        
    Returns:
    --------
    torch.Tensor
        Fairness penalty value
    """
    sensitive_0 = encoded[sensitive == 0]
    sensitive_1 = encoded[sensitive == 1]

    mean_0 = sensitive_0.mean(dim=0) if len(sensitive_0) > 0 else torch.zeros(encoded.shape[1])
    mean_1 = sensitive_1.mean(dim=0) if len(sensitive_1) > 0 else torch.zeros(encoded.shape[1])

    return torch.norm(mean_0 - mean_1, p=2)  # L2 Distance between means
