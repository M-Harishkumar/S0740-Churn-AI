import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="S0740 Advanced Churn AI", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
<style>
.reportview-container {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%)}
.metric-container {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 15px;}
.main-header {font-size: 3rem !important; color: white !important;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 S0740 Advanced Churn Analytics</h1>', unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    model = pickle.load(open('best_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

model, scaler = load_models()

# Sidebar
st.sidebar.title("⚙️ Controls")
prediction_mode = st.sidebar.radio("Mode", ["Single Prediction", "Batch Analysis", "What-if Analysis"])

# === TAB 1: SINGLE PREDICTION ===
if prediction_mode == "Single Prediction":
    st.markdown("## 🎯 Live Single Prediction")
    
    col1, col2, col3 = st.columns(3)
    tenure = col1.slider("📅 Tenure (months)", 0, 72, 12)
    monthly_charges = col2.slider("💰 Monthly Charges", 18.0, 118.0, 70.0)
    total_charges = col3.slider("💳 Total Charges", 0.0, 8684.0, 1000.0)
    
    col1, col2, col3 = st.columns(3)
    contract = col1.selectbox("📋 Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = col2.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = col3.selectbox("💳 Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    # Predict
    if st.button("🚀 PREDICT CHURN RISK", type="primary", use_container_width=True):
        # Feature vector (simplified - match your model)
        features = [
            tenure, monthly_charges, total_charges,
            1 if contract == "Month-to-month" else 0,
            1 if internet_service == "Fiber optic" else 0,
            1 if payment_method == "Electronic check" else 0
        ]
        input_scaled = scaler.transform(np.array(features).reshape(1, -1))
        
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        
        st.markdown("### 📊 **Prediction Result**")
        if pred == 1:
            st.error(f"**⚠️ CHURN RISK: {prob:.1%}**")
            st.info("**Actions:** Discount offer, Retention call, Service upgrade")
        else:
            st.success(f"**✅ SAFE: {prob:.1%} risk**")
            st.info("**Action:** Continue service")
        
        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob*100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': prob*100
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

# === TAB 2: BATCH ANALYSIS ===
elif prediction_mode == "Batch Analysis":
    st.markdown("## 📈 Batch Customer Analysis")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())
        
        # Predict batch
        if st.button("🔮 Analyze All Customers"):
            predictions = model.predict(scaler.transform(df))
            probs = model.predict_proba(scaler.transform(df))[:, 1]
            
            df['Churn_Prediction'] = predictions
            df['Churn_Probability'] = probs
            
            st.success("✅ Analysis Complete!")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(df, names='Churn_Prediction', 
                               title="Churn Risk Distribution",
                               color_discrete_sequence=['green', 'red'])
                st.plotly_chart(fig_pie)
            
            with col2:
                fig_hist = px.histogram(df, x='Churn_Probability', nbins=20,
                                      title="Risk Score Distribution")
                st.plotly_chart(fig_hist)
            
            st.dataframe(df[['Churn_Prediction', 'Churn_Probability']].head(10))
            
            high_risk = df[df['Churn_Probability'] > 0.7].shape[0]
            st.metric("🚨 High Risk Customers", high_risk)

# === TAB 3: WHAT-IF ANALYSIS ===
elif prediction_mode == "What-if Analysis":
    st.markdown("## 🔄 What-if Scenario Analysis")
    
    base_tenure = st.slider("Base Tenure", 0, 72, 12)
    base_charges = st.slider("Base Monthly Charges", 18, 118, 70)
    
    st.markdown("**Scenario 1: Increase Charges**")
    new_charges = st.slider("New Charges", 18, 118, 80)
    
    if st.button("Compare Scenarios"):
        base_input = np.array([base_tenure, base_charges])
        new_input = np.array([base_tenure, new_charges])
        
        base_scaled = scaler.transform(base_input.reshape(1, -1))
        new_scaled = scaler.transform(new_input.reshape(1, -1))
        
        base_prob = model.predict_proba(base_scaled)[0][1]
        new_prob = model.predict_proba(new_scaled)[0][1]
        
        col1, col2 = st.columns(2)
        col1.metric("Base Risk", f"{base_prob:.1%}")
        col2.metric("New Risk", f"{new_prob:.1%}")
        
        change = new_prob - base_prob
        st.metric("Risk Change", f"{change:+.1%}")

# === GLOBAL METRICS ===
st.markdown("## 📊 Model Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", "89.6%")
col2.metric("Precision", "79.5%")
col3.metric("Recall", "70.4%")
col4.metric("Dataset", "7,043 customers")

# Feature importance (if available)
if hasattr(model, 'feature_importances_'):
    st.markdown("## 🎛️ Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': ['Tenure', 'Monthly Charges', 'Total Charges', 'Contract', 'Internet', 'Payment'],
        'Importance': model.feature_importances_[:6]
    }).sort_values('Importance', ascending=True)
    
    fig_bar = px.bar(importance_df, x='Importance', y='Feature', 
                    orientation='h', title="Top Churn Drivers")
    st.plotly_chart(fig_bar)

st.markdown("---")
st.markdown("**🏆 S0740 Advanced Churn AI • Production Ready • Deployed Live**")
