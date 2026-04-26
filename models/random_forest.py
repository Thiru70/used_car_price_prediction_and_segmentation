"""
models/random_forest.py
-----------------------
Random Forest Regressor for Used Car Price Prediction.

Functions:
    train_random_forest(X_train, y_train) -> model
    predict_random_forest(model, X)       -> np.ndarray
"""

from sklearn.ensemble import RandomForestRegressor


def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
):
    """
    Train a Random Forest Regressor (ensemble of Decision Trees).

    Parameters
    ----------
    X_train      : array-like, shape (n_samples, n_features)
    y_train      : array-like, shape (n_samples,)
    n_estimators : number of trees in the forest
    max_depth    : maximum depth per tree
    random_state : seed for reproducibility

    Returns
    -------
    Fitted RandomForestRegressor model
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,        # use all CPU cores
    )
    model.fit(X_train, y_train)
    print(f"[Random Forest] Training complete. Trees: {n_estimators}")
    return model


def predict_random_forest(model, X):
    """
    Generate price predictions using a trained Random Forest model.

    Parameters
    ----------
    model : fitted RandomForestRegressor
    X     : array-like, shape (n_samples, n_features)

    Returns
    -------
    np.ndarray of predicted prices
    """
    predictions = model.predict(X)
    return predictions
