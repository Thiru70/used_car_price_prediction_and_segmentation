"""
models/knn.py
-------------
K-Nearest Neighbors Regressor for Used Car Price Prediction.

Functions:
    train_knn(X_train, y_train) -> model
    predict_knn(model, X)       -> np.ndarray
"""

from sklearn.neighbors import KNeighborsRegressor


def train_knn(X_train, y_train, n_neighbors: int = 7, weights: str = "distance"):
    """
    Train a K-Nearest Neighbors Regressor.

    Parameters
    ----------
    X_train     : array-like, shape (n_samples, n_features)
                  ** Must be scaled — KNN is distance-based **
    y_train     : array-like, shape (n_samples,)
    n_neighbors : number of nearest neighbours to consider
    weights     : 'uniform'  → all neighbours equal weight
                  'distance' → closer neighbours weigh more

    Returns
    -------
    Fitted KNeighborsRegressor model
    """
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        metric="euclidean",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"[KNN] Training complete. k={n_neighbors}, weights='{weights}'")
    return model


def predict_knn(model, X):
    """
    Generate price predictions using a trained KNN model.

    Parameters
    ----------
    model : fitted KNeighborsRegressor
    X     : array-like, shape (n_samples, n_features) — must be scaled

    Returns
    -------
    np.ndarray of predicted prices
    """
    predictions = model.predict(X)
    return predictions
