import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title="IoT Predictive Maintenance", layout="wide")

st.title("🔧 IoT Predictive Maintenance — NASA Turbofan Sensor Analytics")
st.markdown("""
This dashboard presents sensor data analytics on NASA's C-MAPSS turbofan 
engine dataset, predicting **Remaining Useful Life (RUL)** for predictive 
maintenance.
""")

# --- Metrics ---
st.header("📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE (raw sensors)", "17.89")
col2.metric("MAE (raw sensors)", "13.05")
col3.metric("RMSE (+ rolling features)", "19.12")
col4.metric("MAE (+ rolling features)", "13.94")

st.divider()

# --- Sensor Trends ---
st.header("📈 Sensor Degradation Trends")
st.image("results/sensor_trends_engine1.png", caption="Sensor readings over engine lifecycle")

# --- Correlation Heatmap ---
st.header("🔍 Sensor Correlation Heatmap")
st.image("results/correlation_heatmap.png", caption="Correlation between sensors and RUL")

# --- Prediction Accuracy ---
st.header("🎯 Predicted vs Actual RUL")
st.image("results/predicted_vs_actual.png", caption="Model prediction accuracy")

# --- Maintenance Alerts ---
st.header("🚨 Maintenance Alert Dashboard")
st.image("results/maintenance_alerts.png", caption="Engines flagged for maintenance")

st.subheader("Alert Table")
try:
    alert_df = pd.read_csv("results/maintenance_alerts.csv")
    st.dataframe(alert_df.sort_values("predicted_RUL"), use_container_width=True)
except FileNotFoundError:
    st.warning("Alert data not found — upload maintenance_alerts.csv to /results")

st.divider()
st.markdown("Built with Python, scikit-learn, and Streamlit | [View source on GitHub](https://github.com/Manishkv9/iot-predictive-maintenance)")
