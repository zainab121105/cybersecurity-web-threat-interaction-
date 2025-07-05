import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

# ---------------------------
# PAGE CONFIGURATION
# ---------------------------
st.set_page_config(page_title="Web Threat Detector", layout="wide")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cyber.csv")
    df['creation_time'] = pd.to_datetime(df['creation_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    df['time'] = pd.to_datetime(df['time'])
    df['duration_seconds'] = (df['end_time'] - df['creation_time']).dt.total_seconds()
    df['avg_packet_size'] = (df['bytes_in'] + df['bytes_out']) / df['duration_seconds']
    df['log_bytes_in'] = np.log1p(df['bytes_in'])
    return df

df = load_data()

# ---------------------------
# TITLE & DESCRIPTION
# ---------------------------
st.title("🔐 Suspicious Web Threat Interactions")
st.markdown("Analyze web traffic to detect potentially malicious or suspicious activity using anomaly detection.")

# ---------------------------
# FILTERS
# ---------------------------
with st.sidebar:
    st.header("🔎 Filter")
    countries = st.multiselect("Source Country", df['src_ip_country_code'].dropna().unique())
    if countries:
        df = df[df['src_ip_country_code'].isin(countries)]

# ---------------------------
# SHOW RAW DATA
# ---------------------------
if st.checkbox("Show Raw Data"):
    st.dataframe(df.head(10))

# ---------------------------
# HISTOGRAM: Bytes In (Log Scale)
# ---------------------------
st.subheader("📊 Log-Scaled Bytes In Distribution")
fig, ax = plt.subplots()
sns.histplot(df['log_bytes_in'], bins=50, kde=True, color='blue', ax=ax)
plt.xlabel("Log(Bytes In)")
st.pyplot(fig)

# ---------------------------
# ANOMALY DETECTION
# ---------------------------
st.subheader("⚠️ Anomaly Detection using Isolation Forest")

features = df[['bytes_in', 'bytes_out', 'duration_seconds', 'avg_packet_size']].fillna(0)

model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(features)
df['anomaly'] = df['anomaly'].map({-1: 'Suspicious', 1: 'Normal'})

suspicious_df = df[df['anomaly'] == 'Suspicious']
normal_df = df[df['anomaly'] == 'Normal']

st.markdown(f"**Suspicious Sessions Detected:** {len(suspicious_df)} out of {len(df)}")

# ---------------------------
# SHOW ANOMALIES
# ---------------------------
st.dataframe(suspicious_df[['src_ip', 'dst_ip', 'src_ip_country_code', 'bytes_in', 'bytes_out', 'duration_seconds']].head(10))

# ---------------------------
# DOWNLOAD BUTTON
# ---------------------------
csv = suspicious_df.to_csv(index=False)
st.download_button("📥 Download Suspicious Records", csv, "suspicious_traffic.csv", "text/csv")
