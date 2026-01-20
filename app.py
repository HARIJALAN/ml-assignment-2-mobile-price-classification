import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Mobile Price Classification",
    layout="centered"
)

st.title("Mobile Price Classification")
st.write("Machine Learning Assignment – Classification Models")

# =========================
# Load trained models
# =========================
models, results, scaler = pickle.load(
    open("model/saved_models.pkl", "rb")
)

# =========================
# Convert results to DataFrame (ALL OBSERVATIONS)
# =========================
results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={"index": "Model"}, inplace=True)

# =========================
# Download ALL observations
# =========================
st.subheader("Download Model Evaluation Results")

csv_data = results_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download All Model Metrics (CSV)",
    data=csv_data,
    file_name="model_evaluation_results.csv",
    mime="text/csv"
)

# =========================
# Model selection
# =========================
st.subheader("Model-wise Analysis")

model_name = st.selectbox(
    "Select Model",
    list(models.keys())
)
model = models[model_name]

# Show metrics for selected model
st.subheader("Evaluation Metrics (Selected Model)")
st.json(results[model_name])

# =========================
# Upload test dataset
# =========================
uploaded_file = st.file_uploader(
    "Upload Test CSV (without price_range column)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    X = scaler.transform(df)
    preds = model.predict(X)

    st.subheader("Predictions")
    st.write(preds)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(preds, preds)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
