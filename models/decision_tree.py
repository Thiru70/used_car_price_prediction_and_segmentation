"""
models/decision_tree.py
-----------------------
Decision Tree Regressor for Used Car Price Prediction.

Functions:
    train_decision_tree(X_train, y_train) -> model
    predict_decision_tree(model, X)       -> np.ndarray
"""

from sklearn.tree import DecisionTreeRegressor


def train_decision_tree(X_train, y_train, max_depth: int = 8, random_state: int = 42):
    """
    Train a Decision Tree Regressor.

    Parameters
    ----------
    X_train      : array-like, shape (n_samples, n_features)
    y_train      : array-like, shape (n_samples,)
    max_depth    : maximum depth of the tree (controls over-fitting)
    random_state : seed for reproducibility

    Returns
    -------
    Fitted DecisionTreeRegressor model
    """
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=random_state,
        min_samples_split=10,   # avoid splitting tiny leaves
        min_samples_leaf=5
    )
    model.fit(X_train, y_train)
    print("[Decision Tree] Training complete.")
    return model


def predict_decision_tree(model, X):
    """
    Generate price predictions using a trained Decision Tree model.

    Parameters
    ----------
    model : fitted DecisionTreeRegressor
    X     : array-like, shape (n_samples, n_features)

    Returns
    -------
    np.ndarray of predicted prices
    """
    predictions = model.predict(X)
    return predictions
