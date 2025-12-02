import numpy as np
import time
from backend.data_loader import get_stock_data
from backend.feature_engineering import create_features
from backend.models.gradient_boosting import GradientBoostingRegressorCustom
from backend.models.random_forest import RandomForestRegressorCustom
from backend.models.arima import arima_forecast
from backend.models.kalman_filter import kalman_filter_numba
from backend.models.q_learning import q_learning_episode_numba
from backend.utils.performance import compute_performance


def run_experiment(ticker="AAPL", model_name="Gradient Boosting", episodes=10, verbose=True):
    """
    Runs only the selected model and prints progress to help debug.
    """

    # ============ HEADER ============
    if verbose:
        print(f"\n{'='*70}")
        print(f" Starting Algorithmic Trading Experiment")
        print(f" Ticker: {ticker} |  Model: {model_name}")
        print(f"{'='*70}\n")

    start_time = time.time()

    # =======================
    #  Data Loading
    # =======================
    if verbose: print("Loading stock data...")
    try:
        df = get_stock_data(ticker)
    except Exception as e:
        error_msg = f" Failed to load stock data for {ticker}: {str(e)}"
        if verbose:
            print(error_msg)
        raise ValueError(error_msg)
    
    # Validate data is not empty
    if df is None or len(df) == 0:
        error_msg = f" No data loaded for {ticker}. Please check the ticker symbol and try again."
        if verbose:
            print(error_msg)
        raise ValueError(error_msg)
    
    if verbose: print(f" Loaded {len(df)} rows of data.\n")

    # =======================
    # 2️ Feature Engineering
    # =======================
    if verbose: print(" Creating features...")
    df_feat = create_features(df)
    if verbose: print(f" Feature matrix created with {df_feat.shape[1]} columns.\n")

    # Split data
    X = df_feat.drop(columns=['Target']).select_dtypes(include=[np.number]).values
    y = df_feat['Target'].values
    
    # Validate we have enough data
    if len(X) < 10:
        error_msg = f" Insufficient data after feature engineering: {len(X)} rows. Need at least 10 rows."
        if verbose:
            print(error_msg)
        raise ValueError(error_msg)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Validate train/test splits
    if len(X_train) == 0 or len(X_test) == 0:
        error_msg = f" Insufficient data for train/test split: train={len(X_train)}, test={len(X_test)}"
        if verbose:
            print(error_msg)
        raise ValueError(error_msg)
    
    # Get feature names for explainability
    feature_names = df_feat.drop(columns=['Target']).select_dtypes(include=[np.number]).columns.tolist()

    # Base result structure
    result = {
        "model_name": model_name,
        "y_test": y_test,
        "y_pred": None,
        "eq": None,
        "performance": None,
        "model": None,  # Store trained model for explainability
        "X_train": X_train,  # Store training data for explainability
        "feature_names": feature_names  # Store feature names for explainability
    }

    # =======================
    # 3️ Model Execution
    # =======================

    # --- Gradient Boosting ---
    if model_name == "Gradient Boosting":
        if verbose: print(" Training Gradient Boosting Model...")
        model = GradientBoostingRegressorCustom()
        model.fit(X_train, y_train)
        if verbose: print(" Training complete. Predicting...")
        y_pred = model.predict(X_test)
        result["y_pred"] = y_pred
        result["model"] = model  # Store model for explainability
        # Calculate metrics with proper NaN handling
        if len(y_test) == 0 or len(y_pred) == 0:
            rmse, r2 = float('nan'), float('nan')
        else:
            rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
            # R² calculation with division by zero protection
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            if ss_tot == 0 or np.isnan(ss_tot):
                r2 = float('nan')
            else:
                r2 = float(1 - ss_res / ss_tot)
        result["performance"] = {"RMSE": rmse, "R2": r2}
        if not (np.isnan(rmse) or np.isnan(r2)):
            print(f" RMSE: {rmse:.3f} | R²: {r2:.3f}")
        else:
            print(f" RMSE: {rmse} | R²: {r2} (NaN - check data quality)")

    # --- Random Forest ---
    elif model_name == "Random Forest":
        if verbose: print(" Training Random Forest Model...")
        model = RandomForestRegressorCustom()
        model.fit(X_train, y_train)
        if verbose: print(" Training complete. Predicting...")
        y_pred = model.predict(X_test)
        result["y_pred"] = y_pred
        result["model"] = model  # Store model for explainability
        # Calculate metrics with proper NaN handling
        if len(y_test) == 0 or len(y_pred) == 0:
            rmse, r2 = float('nan'), float('nan')
        else:
            rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
            # R² calculation with division by zero protection
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            if ss_tot == 0 or np.isnan(ss_tot):
                r2 = float('nan')
            else:
                r2 = float(1 - ss_res / ss_tot)
        result["performance"] = {"RMSE": rmse, "R2": r2}
        if not (np.isnan(rmse) or np.isnan(r2)):
            print(f" RMSE: {rmse:.3f} | R²: {r2:.3f}")
        else:
            print(f" RMSE: {rmse} | R²: {r2} (NaN - check data quality)")

    # --- ARIMA ---
    elif model_name == "ARIMA":
        if verbose: print(" Running ARIMA Forecast...")
        y_pred = arima_forecast(df_feat['Close'].values, len(y_test))
        result["y_pred"] = y_pred
        # Calculate metrics with proper NaN handling
        if len(y_test) == 0 or len(y_pred) == 0:
            rmse, r2 = float('nan'), float('nan')
        else:
            rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
            # R² calculation with division by zero protection
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            if ss_tot == 0 or np.isnan(ss_tot):
                r2 = float('nan')
            else:
                r2 = float(1 - ss_res / ss_tot)
        result["performance"] = {"RMSE": rmse, "R2": r2}
        if not (np.isnan(rmse) or np.isnan(r2)):
            print(f" RMSE: {rmse:.3f} | R²: {r2:.3f}")
        else:
            print(f" RMSE: {rmse} | R²: {r2} (NaN - check data quality)")

    # --- Q-Learning Agent ---
    elif model_name == "Q-Learning Agent":
        if verbose:
            print(f" Running Q-Learning Agent for {episodes} episodes...")

        df_rl = df_feat.copy()
        Q = np.zeros((3, 3, 3, 3))  # 3 states (s0, s1, s2), 3 actions

        # --- Training phase ---
        for ep in range(1, episodes + 1):
            Q, _ = q_learning_episode_numba(
                df_rl['Close'].values,
                df_rl['MA20'].values,
                df_rl['MA50'].values,
                df_rl['RSI'].values,
                Q, 0.05, 0.98, 0.2  # α, γ, ε
            )
            if verbose and ep % max(1, episodes // 5) == 0:
                print(f"    Episode {ep}/{episodes} complete.")

        # --- Testing / Deployment phase (no exploration) ---
        _, eq = q_learning_episode_numba(
            df_rl['Close'].values,
            df_rl['MA20'].values,
            df_rl['MA50'].values,
            df_rl['RSI'].values,
            Q, 0.05, 0.98, 0.0  # ε=0 for deterministic run
        )

        # --- Compute performance ---
        perf = compute_performance(eq)
        result["eq"] = eq
        result["performance"] = {k: float(v) for k, v in perf.items()}

        print(f"\n Final Q-Agent Performance:")
        for k, v in perf.items():
            print(f"   - {k}: {v:.4f}")



    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # ============ DONE ============
    duration = time.time() - start_time
    if verbose:
        print(f"\n Completed in {duration:.2f} seconds.")
        print(f"{'='*70}\n")
    
    # Debug: Verify model is stored (for explainability)
    if verbose and result.get("model") is not None:
        print(f" Model stored successfully for explainability.")
        print(f"   Model type: {type(result['model']).__name__}")
    elif result.get("model") is None and model_name in ["Gradient Boosting", "Random Forest"]:
        print(f" WARNING: Model not stored! This will break explainability.")
    
    return result


# -------------------------------
# Manual Test Run
# -------------------------------
if __name__ == "__main__":
    out = run_experiment("AAPL", model_name="Q-Learning Agent", episodes=5)
    print("\n Final Performance Metrics:")
    print(out["performance"])
