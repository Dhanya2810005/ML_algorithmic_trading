# backend/utils/explainable_ai.py
"""
Explainable AI helpers using SHAP and LIME.
These functions accept trained models and data as parameters and return HTML visualizations
that can be embedded in Streamlit via st.components.v1.html(...).

If SHAP/LIME aren't installed, functions return None and the frontend shows placeholders.
"""

import io
import base64
import numpy as np

def _safe_import(name):
    try:
        module = __import__(name)
        return module
    except Exception:
        return None

shap = _safe_import("shap")
lime = _safe_import("lime")
matplotlib = _safe_import("matplotlib")
plotly = _safe_import("plotly")

def get_shap_summary_html(model, X_train, feature_names=None, max_samples=100):
    """
    Compute global SHAP summary plot and return HTML string.
    
    Args:
        model: Trained model with .predict() method
        X_train: Training feature matrix (numpy array or similar)
        feature_names: List of feature names (optional, defaults to f0, f1, ...)
        max_samples: Maximum number of samples to use for SHAP (for performance)
    
    Returns:
        HTML string with embedded image, or None if SHAP unavailable or error occurs
    """
    try:
        if shap is None:
            return None
        
        if model is None or X_train is None:
            return None
        
        # Limit samples for performance
        if len(X_train) > max_samples:
            indices = np.random.choice(len(X_train), max_samples, replace=False)
            X_sample = X_train[indices] if isinstance(X_train, np.ndarray) else X_train.iloc[indices].values
        else:
            X_sample = X_train if isinstance(X_train, np.ndarray) else X_train.values
        
        # Ensure numpy array
        X_sample = np.asarray(X_sample, dtype=np.float64)
        
        # Try different SHAP explainers based on model type
        # Note: Our custom tree models won't work with TreeExplainer (requires sklearn-style trees)
        # So we'll use KernelExplainer or general Explainer which work with any predict() method
        explainer = None
        try:
            # First try: General Explainer (works with any model that has predict method)
            explainer = shap.Explainer(model.predict, X_sample[:min(50, len(X_sample))])
        except Exception:
            try:
                # Fallback: KernelExplainer (slower but more general, works with any predict method)
                explainer = shap.KernelExplainer(model.predict, X_sample[:min(50, len(X_sample))])
            except Exception:
                # Last resort: try TreeExplainer (won't work with our custom trees but worth trying)
                try:
                    explainer = shap.TreeExplainer(model)
                except Exception:
                    return None
        
        # Compute SHAP values
        shap_values = explainer(X_sample[:min(100, len(X_sample))])
        
        # Create matplotlib plot
        if matplotlib is None:
            return None
            
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        
        # Use feature names if provided
        if feature_names is not None:
            shap.summary_plot(shap_values, X_sample[:min(100, len(X_sample))], 
                            feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, X_sample[:min(100, len(X_sample))], show=False)
        
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        html = f"<img src='data:image/png;base64,{img_b64}' style='width:100%;'/>"
        return html
    except Exception as e:
        # Return None on any error (will be caught by frontend)
        return None

def get_lime_html_for_sample(model, X_train, sample_index=0, feature_names=None):
    """
    Run LIME local explanation for a specific sample and return HTML snippet.
    
    Args:
        model: Trained model with .predict() method
        X_train: Training feature matrix (numpy array or similar)
        sample_index: Index of sample to explain (default: 0)
        feature_names: List of feature names (optional)
    
    Returns:
        HTML string with LIME explanation, or None if LIME unavailable or error occurs
    """
    try:
        if lime is None:
            return None
        
        if model is None or X_train is None:
            return None
        
        # Ensure numpy array
        X_array = X_train if isinstance(X_train, np.ndarray) else X_train.values
        X_array = np.asarray(X_array, dtype=np.float64)
        
        # Ensure sample_index is valid
        sample_index = min(sample_index, len(X_array) - 1)
        sample_index = max(0, sample_index)
        
        # Get feature names
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X_array.shape[1])]
        
        # Build LIME explainer
        from lime import lime_tabular
        explainer = lime_tabular.LimeTabularExplainer(
            X_array, 
            mode='regression', 
            feature_names=feature_names,
            discretize_continuous=False
        )
        
        # Explain instance
        exp = explainer.explain_instance(
            X_array[sample_index], 
            model.predict, 
            num_features=min(10, X_array.shape[1])
        )
        
        # Return HTML
        html = exp.as_html()
        return html
    except Exception as e:
        # Return None on any error
        return None

# Legacy function names for backward compatibility
def get_shap_summary_html_if_possible():
    """Legacy function - returns None. Use get_shap_summary_html() with model and data."""
    return None

def get_lime_html_for_sample_if_possible(sample_index=0):
    """Legacy function - returns None. Use get_lime_html_for_sample() with model and data."""
    return None
