"""
app.py
------
Main Flask application for Used Car Price Prediction & Segmentation.

Routes:
    GET  /          → Home page
    GET  /train     → Train all models, display metrics
    GET  /analysis  → Show EDA & visualisation graphs
    POST /predict   → Predict price + cluster for a single car
"""

import os, io, base64, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import _tree
from dotenv import load_dotenv
from supabase import create_client, Client

# ── Local imports ─────────────────────────────────────────────────────────────
from preprocessing import preprocess, preprocess_single
from evaluation    import evaluate_model, compare_models

from models.decision_tree import train_decision_tree, predict_decision_tree
from models.random_forest  import train_random_forest,  predict_random_forest
from models.knn            import train_knn,             predict_knn
from models.kmeans         import train_kmeans,          predict_cluster, get_segment_map, get_cluster_label

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "used_cars.csv")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Supabase Setup ────────────────────────────────────────────────────────────
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── In-memory model store (populated by /train) ───────────────────────────────
trained = {
    "dt_model":       None,
    "rf_model":       None,
    "knn_model":      None,
    "kmeans_model":   None,
    "segment_map":    None,
    "results":        [],
    "feature_cols":   [],
    "df":             None,
    "X_test":         None,
    "y_test":         None,
}


# ── Helper: encode matplotlib figure → base64 PNG string ─────────────────────
def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE: Home
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/")
def home():
    """Landing page — explains the project."""
    return render_template("index.html")


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE: Train
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/train")
def train():
    """Train all four models and display performance metrics."""
    global trained

    # 1. Preprocess
    X_scaled, y, df = preprocess(DATASET_PATH)
    feature_cols = [c for c in df.columns if c != "Selling_Price"]

    # 2. Train / test split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 3. Train supervised models
    dt_model  = train_decision_tree(X_train, y_train)
    rf_model  = train_random_forest(X_train, y_train)
    knn_model = train_knn(X_train, y_train)

    # 4. Train K-Means on ALL scaled features (unsupervised)
    kmeans_model = train_kmeans(X_scaled)
    segment_map  = get_segment_map(kmeans_model, y)

    # 5. Evaluate supervised models on test set
    results = compare_models([
        evaluate_model("Decision Tree", y_test, predict_decision_tree(dt_model, X_test)),
        evaluate_model("Random Forest", y_test, predict_random_forest(rf_model, X_test)),
        evaluate_model("KNN",           y_test, predict_knn(knn_model, X_test)),
    ])

    # 6. Persist models in memory
    trained.update({
        "dt_model":     dt_model,
        "rf_model":     rf_model,
        "knn_model":    knn_model,
        "kmeans_model": kmeans_model,
        "segment_map":  segment_map,
        "results":      results,
        "feature_cols": feature_cols,
        "df":           df,
        "X_test":       X_test,
        "y_test":       y_test,
    })

    # 7. Save models to disk with pickle
    for name, obj in [("dt", dt_model), ("rf", rf_model),
                      ("knn", knn_model), ("kmeans", kmeans_model)]:
        with open(os.path.join(MODELS_DIR, f"{name}_model.pkl"), "wb") as f:
            pickle.dump(obj, f)

    # 8. Build comparison bar chart
    model_names = [r["model"] for r in results]
    r2_values   = [r["R2"]    for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#6C63FF", "#00D4AA", "#FF6B6B"]
    bars = ax.bar(model_names, r2_values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Model Comparison — R² Score", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines[:].set_edgecolor("#444")
    chart_b64 = fig_to_base64(fig)

    return render_template(
        "train.html",
        results=results,
        chart=chart_b64,
        trained=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE: Analysis (EDA & Visualisations)
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/analysis")
def analysis():
    """Generate and display three EDA charts."""
    if trained["df"] is None:
        return render_template("analysis.html", error="Please train the models first.")

    df          = trained["df"]
    kmeans_model = trained["kmeans_model"]

    # ── Chart 1: Selling Price vs Year ───────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    year_price = df.groupby("Year")["Selling_Price"].median().reset_index()
    ax1.plot(year_price["Year"], year_price["Selling_Price"] / 1e5,
             color="#6C63FF", linewidth=2.5, marker="o", markersize=5)
    ax1.fill_between(year_price["Year"], year_price["Selling_Price"] / 1e5,
                     alpha=0.15, color="#6C63FF")
    ax1.set_xlabel("Year", color="white")
    ax1.set_ylabel("Median Selling Price (₹ Lakhs)", color="white")
    ax1.set_title("Selling Price Trend by Year", fontsize=14, fontweight="bold", color="white")
    _style_dark_ax(ax1, fig1)
    chart1 = fig_to_base64(fig1)

    # ── Chart 2: Correlation Heatmap ─────────────────────────────────────────
    numeric_df = df.select_dtypes(include=[np.number])
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap=cmap,
                ax=ax2, linewidths=0.5, linecolor="#222",
                annot_kws={"size": 8}, vmin=-1, vmax=1)
    ax2.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", color="white")
    ax2.tick_params(colors="white", labelsize=8)
    fig2.patch.set_facecolor("#1a1a2e")
    ax2.set_facecolor("#16213e")
    chart2 = fig_to_base64(fig2)

    # ── Chart 3: K-Means Cluster Scatter ─────────────────────────────────────
    # Use Year vs Selling_Price (both already encoded/numeric in df)
    segment_map = trained["segment_map"]
    raw_labels  = kmeans_model.labels_
    seg_labels  = [get_cluster_label(c, segment_map) for c in raw_labels]
    palette = {"Budget": "#00D4AA", "Mid-Range": "#6C63FF", "Premium": "#FF6B6B"}

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for seg in set(seg_labels):
        mask = [s == seg for s in seg_labels]
        ax3.scatter(
            df["Year"][mask],
            df["Selling_Price"][mask] / 1e5,
            label=seg,
            alpha=0.65,
            s=18,
            color=palette.get(seg, "#aaa"),
        )
    ax3.set_xlabel("Year", color="white")
    ax3.set_ylabel("Selling Price (₹ Lakhs)", color="white")
    ax3.set_title("K-Means Market Segmentation", fontsize=14, fontweight="bold", color="white")
    legend = ax3.legend(facecolor="#16213e", labelcolor="white", fontsize=9)
    _style_dark_ax(ax3, fig3)
    chart3 = fig_to_base64(fig3)

    return render_template("analysis.html", chart1=chart1, chart2=chart2, chart3=chart3, error=None)


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE: API Analysis Data
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/api/analysis-data")
def api_analysis_data():
    """Returns JSON data for interactive dashboard visualisations."""
    if trained["df"] is None:
        return jsonify({"error": "Please train models first"}), 400

    X_test = trained["X_test"]
    y_test = trained["y_test"]
    df = trained["df"]
    feature_cols = trained["feature_cols"]

    # 1. PCA for 2D Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)
    
    # K-Means clusters for the test set
    kmeans_model = trained["kmeans_model"]
    cluster_ids = kmeans_model.predict(X_test)
    segment_map = trained["segment_map"]
    
    # 2. Extract Tree Structure (Decision Tree)
    dt_model = trained["dt_model"]
    tree_data = _export_tree_to_json(dt_model, feature_cols)

    # 3. Extract Random Forest (first 3 trees)
    rf_model = trained["rf_model"]
    forest_data = [
        _export_tree_to_json(est, feature_cols) 
        for est in rf_model.estimators_[:3]
    ]

    # 4. KNN Data (Points + Classes)
    # Map prices to segments for classification view
    def get_seg(price):
        if price < 400000: return "Budget"
        if price < 1000000: return "Mid-Range"
        return "Premium"
    
    y_segments = [get_seg(p) for p in y_test]

    data = {
        "pca_points": [
            {"x": float(p[0]), "y": float(p[1]), "cluster": segment_map.get(c), "actual_segment": s}
            for p, c, s in zip(X_pca, cluster_ids, y_segments)
        ],
        "tree": tree_data,
        "forest": forest_data,
        "centroids": pca.transform(kmeans_model.cluster_centers_).tolist()
    }
    return jsonify(data)


def _export_tree_to_json(model, feature_names):
    """Recursively export sklearn decision tree to JSON format for D3.js."""
    tree_ = model.tree_
    
    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            left = recurse(tree_.children_left[node], depth + 1)
            right = recurse(tree_.children_right[node], depth + 1)
            return {
                "name": name,
                "threshold": round(float(threshold), 2),
                "children": [left, right]
            }
        else:
            return {
                "name": "Price",
                "value": round(float(tree_.value[node][0][0]), 0)
            }

    return recurse(0, 1)


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE: Predict
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Accept car details via a form and return price + segment predictions."""
    prediction_result = None

    if request.method == "POST":
        if trained["dt_model"] is None:
            return render_template("predict.html",
                                   error="Models not trained yet. Please visit /train first.")

        # Collect form data
        input_data = {
            "Year":         int(request.form.get("year", 2018)),
            "Brand":        request.form.get("brand", "Maruti"),
            "Fuel_Type":    request.form.get("fuel_type", "Petrol"),
            "Transmission": request.form.get("transmission", "Manual"),
            "Owner":        request.form.get("owner", "First Owner"),
            "KM_Driven":    float(request.form.get("km_driven", 50000)),
            "Mileage":      float(request.form.get("mileage", 18.0)),
            "Engine":       int(request.form.get("engine", 1200)),
            "Max_Power":    float(request.form.get("max_power", 80.0)),
            "Seats":        int(request.form.get("seats", 5)),
        }

        feature_cols = trained["feature_cols"]
        X_input = preprocess_single(input_data, feature_cols)

        # Predictions from all three regressors
        dt_price  = float(predict_decision_tree(trained["dt_model"],  X_input)[0])
        rf_price  = float(predict_random_forest(trained["rf_model"],  X_input)[0])
        knn_price = float(predict_knn(trained["knn_model"], X_input)[0])
        avg_price = (dt_price + rf_price + knn_price) / 3

        # K-Means cluster
        cluster_id = predict_cluster(trained["kmeans_model"], X_input)[0]
        segment    = get_cluster_label(cluster_id, trained["segment_map"])

        prediction_result = {
            "dt_price":  f"₹ {dt_price:,.0f}",
            "rf_price":  f"₹ {rf_price:,.0f}",
            "knn_price": f"₹ {knn_price:,.0f}",
            "avg_price": f"₹ {avg_price:,.0f}",
            "segment":   segment,
            "input":     input_data,
        }

        # Save to Supabase History
        try:
            history_data = {
                "year":         input_data["Year"],
                "brand":        input_data["Brand"],
                "fuel_type":    input_data["Fuel_Type"],
                "transmission": input_data["Transmission"],
                "owner":        input_data["Owner"],
                "km_driven":    input_data["KM_Driven"],
                "mileage":      input_data["Mileage"],
                "engine":       input_data["Engine"],
                "max_power":    input_data["Max_Power"],
                "seats":        input_data["Seats"],
                "predicted_price": avg_price,
                "segment":      segment
            }
            supabase.table("prediction_history").insert(history_data).execute()
        except Exception as e:
            print(f"Error saving to Supabase: {e}")

    return render_template("predict.html", result=prediction_result, error=None)


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE: History
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/history")
def history():
    """Fetch and display prediction history from Supabase."""
    try:
        response = supabase.table("prediction_history").select("*").order("created_at", desc=True).execute()
        history_list = response.data
    except Exception as e:
        print(f"Error fetching history: {e}")
        history_list = []
    
    return render_template("history.html", history=history_list)


# ═════════════════════════════════════════════════════════════════════════════
# Helper: apply dark theme to a matplotlib Axes
# ═════════════════════════════════════════════════════════════════════════════
def _style_dark_ax(ax, fig):
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("white")
    ax.spines[:].set_edgecolor("#444")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
