# ================= IMPORT LIBRARIES =================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="centered"
)

# ================= CSS =================
st.markdown("""
<style>
.stApp {
    background-color: #000000;
    color: white;
}

h1, h2, h3 {
    color: white;
}

.card {
    background-color: #111111;
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0px 4px 15px rgba(255,255,255,0.05);
}

.footer {
    text-align: center;
    color: #aaaaaa;
    font-size: 13px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown("<h1>🛍️ Mall Customer Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<p>Hierarchical Clustering using Machine Learning</p>", unsafe_allow_html=True)

# ================= LOAD DATA =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📂 Dataset Preview")

CSV_FILE = "https://drive.google.com/file/d/1czxTGMNMm6LStI0MgOF4N7eKT-EGYLPg/view?usp=sharing"

if not os.path.exists(CSV_FILE):
    st.error("❌ CSV file not found. Please upload 'Mall_Customers (2).csv' to the repository.")
    st.stop()

df = pd.read_csv(CSV_FILE)
st.dataframe(df.head())
st.markdown("</div>", unsafe_allow_html=True)

# ================= PREPROCESSING =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("⚙️ Data Preprocessing")

df_processed = df.copy()
df_processed.drop("CustomerID", axis=1, inplace=True)
df_processed["Genre"] = LabelEncoder().fit_transform(df_processed["Genre"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_processed)

st.success("Data preprocessing completed")
st.markdown("</div>", unsafe_allow_html=True)

# ================= DENDROGRAM =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🌳 Dendrogram")

fig1, ax1 = plt.subplots(figsize=(10, 6))
dendrogram(linkage(X_scaled, method="ward"), ax=ax1)
ax1.set_title("Dendrogram")
ax1.set_xlabel("Customers")
ax1.set_ylabel("Euclidean Distance")
st.pyplot(fig1)
st.markdown("</div>", unsafe_allow_html=True)

# ================= CLUSTERING =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🔗 Hierarchical Clustering")

n_clusters = st.slider("Select number of clusters", 2, 10, 5)

hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
labels = hc.fit_predict(X_scaled)

df_processed["Cluster"] = labels
st.dataframe(df_processed.head())
st.markdown("</div>", unsafe_allow_html=True)

# ================= VISUALIZATION =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📊 Cluster Visualization")

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=labels
)
ax2.set_xlabel("Annual Income (k$)")
ax2.set_ylabel("Spending Score (1-100)")
ax2.set_title("Customer Segmentation")
st.pyplot(fig2)
st.markdown("</div>", unsafe_allow_html=True)

# ================= EVALUATION =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📈 Model Evaluation")

sil_score = silhouette_score(X_scaled, labels)
st.success(f"Silhouette Score: {sil_score:.4f}")
st.markdown("</div>", unsafe_allow_html=True)

# ================= SAVE MODEL =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("💾 Save Model")

if st.button("Save Model & Scaler"):
    joblib.dump(hc, "hierarchical_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    st.success("Model and Scaler saved successfully!")
st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("<div class='footer'>📘 For academic and internship use only</div>", unsafe_allow_html=True)
