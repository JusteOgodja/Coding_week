import shap
import matplotlib.pyplot as plt
import numpy as np
import joblib

def shap_explainer(model, X_train, X_test, model_name="Model"):
    """
    Generate SHAP explanations for a given model.
    
    Parameters:
        model: Trained model (e.g., LogisticRegression, DecisionTreeClassifier)
        X_train: Training dataset (scaled if necessary)
        X_test: Testing dataset (scaled if necessary)
        model_name: Name of the model (default: "Model")
    """
    # Initialize SHAP explainer
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    
    # Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Summary Plot - {model_name}")
    plt.show()
    
    # Feature Importance Plot
    shap.plots.bar(shap_values, show=False)
    plt.title(f"Feature Importance - {model_name}")
    plt.show()
    
    return shap_values
