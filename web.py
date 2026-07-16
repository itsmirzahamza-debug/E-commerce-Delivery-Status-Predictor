import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="E-commerce Delivery Predictor",
    page_icon="📦",
    layout="wide"
)

# Load the saved model data
@st.cache_resource
def load_model():
    return joblib.load('delivery_model_pipeline.pkl')

try:
    model_data = load_model()
    pipeline = model_data['pipeline']
    categories = model_data['categories']
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    model_loaded = False

# Main Header
st.title("📦 E-commerce Actual Delivery Status Predictor")
st.markdown("""
Predict whether an e-commerce order is likely to be delivered **On Time** or **Late** based on operational and order metrics.
""")

st.write("---")

if model_loaded:
    # Organize layout with columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🛒 Input Order Details")
        st.markdown("Fill in the order specs below to predict delivery performance:")
        
        # Categorical Inputs from our dictionary
        shipping_type = st.selectbox("Shipping Type", categories['shipping_type'])
        customer_segment = st.selectbox("Customer Segment", categories['customer_segment'])
        customer_region = st.selectbox("Customer Region", categories['customer_region'])
        category_name = st.selectbox("Product Category", categories['category_name'])
        
        # Numerical Inputs
        days_scheduled = st.number_input("Scheduled Shipping Days", min_value=0, max_value=30, value=3)
        order_quantity = st.number_input("Order Quantity", min_value=1, max_value=100, value=2)
        sales_per_order = st.number_input("Sales Value ($)", min_value=0.0, max_value=5000.0, value=150.0, step=10.0)
        discount = st.number_input("Discount Value ($)", min_value=0.0, max_value=1000.0, value=10.0, step=1.0)
        profit_per_order = st.number_input("Profit Per Order ($)", min_value=-2000.0, max_value=2000.0, value=25.0, step=5.0)

    with col2:
        st.subheader("🔮 Prediction Analysis")
        
        # Wrap inputs into a temporary dataframe
        input_data = pd.DataFrame([{
            'shipping_type': shipping_type,
            'customer_segment': customer_segment,
            'customer_region': customer_region,
            'category_name': category_name,
            'days_for_shipment_scheduled': days_scheduled,
            'order_item_discount': discount,
            'sales_per_order': sales_per_order,
            'order_quantity': order_quantity,
            'profit_per_order': profit_per_order
        }])
        
        # Display inputs for verification
        with st.expander("🔍 View Raw Features", expanded=False):
            st.dataframe(input_data)
        
        # Trigger Prediction
        if st.button("Run Prediction Pipeline", type="primary", use_container_width=True):
            # Predict & calculate probabilities
            prediction = pipeline.predict(input_data)[0]
            probabilities = pipeline.predict_proba(input_data)[0]
            
            # Display Prediction Card
            st.write("---")
            if prediction == 1:
                st.error("### 🚨 Predicted Status: LATE")
                st.write(f"The model is **{probabilities[1]*100:.2f}%** confident that this delivery will exceed the scheduled schedule.")
            else:
                st.success("### ✅ Predicted Status: ON TIME")
                st.write(f"The model is **{probabilities[0]*100:.2f}%** confident that this delivery will arrive within the scheduled window.")
                
            # Render visual probability bars
            st.write("#### Prediction Confidence Distribution")
            prob_df = pd.DataFrame({
                'Status': ['On Time', 'Late'],
                'Probability': [probabilities[0], probabilities[1]]
            })
            st.bar_chart(prob_df.set_index('Status'))

else:
    st.info("Please make sure 'delivery_model_pipeline.pkl' is in the same directory as this script.")