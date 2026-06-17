import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model('model.h5')
        with open('label_encoder_gender.pkl', 'rb') as f:
            label_encoder_gender = pickle.load(f)
        with open('one_hot_encoder_geo.pkl', 'rb') as f:
            ohe_encoder_geo = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, label_encoder_gender, ohe_encoder_geo, scaler
    except Exception as e:
        st.error(f"❌ Failed to load model artifacts: {e}")
        st.stop()

model, label_encoder_gender, ohe_encoder_geo, scaler = load_artifacts()


st.title("📊 Customer Churn Prediction")
st.markdown("Fill in the customer details below and click **Predict** to assess churn risk.")
st.divider()


st.subheader("🧾 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    geography = st.selectbox("🌍 Geography", ohe_encoder_geo.categories_[0])
    gender = st.selectbox("🧑 Gender", label_encoder_gender.classes_)
    age = st.slider("🎂 Age", 18, 92, 35)

with col2:
    # FIX 3: Input validation — min_value prevents negative/zero inputs
    credit_score = st.number_input("💳 Credit Score", min_value=300, max_value=850, value=650, step=1)
    balance = st.number_input("🏦 Balance", min_value=0.0, value=50000.0, step=100.0, format="%.2f")
    estimated_salary = st.number_input("💰 Estimated Salary", min_value=0.0, value=60000.0, step=100.0, format="%.2f")

with col3:
    tenure = st.slider("📅 Tenure (Years)", 0, 10, 5)
    num_of_products = st.slider("📦 Number of Products", 1, 4, 1)
    has_cr_card = st.selectbox("💳 Has Credit Card", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox("✅ Is Active Member", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.divider()


if st.button("🔍 Predict Churn Risk", use_container_width=True, type="primary"):

    with st.spinner("Analyzing customer data..."):

        # Prepare input dataframe
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        # One-hot encode Geography
        geo_encoded = ohe_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_encoder_geo.get_feature_names_out())

        # Combine and scale
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        input_data_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_data_scaled)
        prediction_proba = float(prediction[0][0])

    st.subheader("📈 Prediction Result")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(
            label="Churn Probability",
            value=f"{prediction_proba * 100:.1f}%"
        )

    with res_col2:
        if prediction_proba >= 0.75:
            st.error("🔴 **High Risk** — This customer is very likely to churn. Immediate retention action recommended.")
        elif prediction_proba >= 0.5:
            st.warning("🟡 **Medium Risk** — This customer may churn. Consider a proactive outreach.")
        else:
            st.success("🟢 **Low Risk** — This customer is not likely to churn.")

    # Visual probability bar
    st.progress(prediction_proba, text=f"Churn Risk: {prediction_proba * 100:.1f}%")