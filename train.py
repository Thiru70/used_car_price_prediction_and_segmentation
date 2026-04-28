"""
train.py
--------
Handles model training, evaluation, and saving.
Can be used independently or imported into Flask app.
"""

import os
import pickle
from sklearn.model_selection import train_test_split

# Local imports
from preprocessing import preprocess
from evaluation import evaluate_model, compare_models

from models.decision_tree import train_decision_tree, predict_decision_tree
from models.random_forest import train_random_forest, predict_random_forest
from models.knn import train_knn, predict_knn
from models.kmeans import train_kmeans, get_segment_map


def train_all_models(dataset_path, models_dir="saved_models"):
    """
    Train all models and return results + trained objects.
    """

    os.makedirs(models_dir, exist_ok=True)

    # 1. Preprocess
    X_scaled, y, df = preprocess(dataset_path)
    feature_cols = [c for c in df.columns if c != "Selling_Price"]

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 3. Train supervised models
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    knn_model = train_knn(X_train, y_train)

    # 4. Train KMeans
    kmeans_model = train_kmeans(X_scaled)
    segment_map = get_segment_map(kmeans_model, y)

    # 5. Evaluate models
    results = compare_models([
        evaluate_model("Decision Tree", y_test, predict_decision_tree(dt_model, X_test)),
        evaluate_model("Random Forest", y_test, predict_random_forest(rf_model, X_test)),
        evaluate_model("KNN", y_test, predict_knn(knn_model, X_test)),
    ])

    # 6. Save models
    model_map = {
        "dt_model.pkl": dt_model,
        "rf_model.pkl": rf_model,
        "knn_model.pkl": knn_model,
        "kmeans_model.pkl": kmeans_model,
    }

    for filename, model in model_map.items():
        with open(os.path.join(models_dir, filename), "wb") as f:
            pickle.dump(model, f)

    return {
        "dt_model": dt_model,
        "rf_model": rf_model,
        "knn_model": knn_model,
        "kmeans_model": kmeans_model,
        "segment_map": segment_map,
        "results": results,
        "feature_cols": feature_cols,
        "df": df,
        "X_test": X_test,
        "y_test": y_test,
    }


# Optional CLI usage
if __name__ == "__main__":
    DATASET_PATH = os.path.join("data", "used_cars.csv")
    output = train_all_models(DATASET_PATH)

    print("\nModel Training Complete!\n")
    for r in output["results"]:
        print(f"{r['model']} → R2: {r['R2']:.4f}")
