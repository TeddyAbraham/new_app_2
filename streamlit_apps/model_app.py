import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .prediction-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-card {
            background: #f0f2f6;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open(r"models\xgb_car_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Main title
st.title("🚗 Car Price Prediction App")
st.markdown("---")
st.write("Get an instant prediction of your car's selling price using our advanced ML model")

model = load_model()

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Enter Car Details")
    
    # Create form for inputs
    with st.form("car_details_form"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            km_driven = st.number_input(
                "Kilometers Driven",
                min_value=0,
                max_value=5000000,
                value=100000,
                step=10000,
                help="Total kilometers the car has been driven"
            )
            
            mileage = st.number_input(
                "Mileage (km/l)",
                min_value=5.0,
                max_value=50.0,
                value=15.0,
                step=0.5,
                help="Fuel efficiency in kilometers per liter"
            )
        
        with col_b:
            age = st.number_input(
                "Car Age (years)",
                min_value=0,
                max_value=50,
                value=5,
                step=1,
                help="How old is the car?"
            )
            
            fuel_type = st.selectbox(
                "Fuel Type",
                ["Petrol", "Diesel", "Electric"],
                help="Select the fuel type of the car"
            )
        
        # Submit button
        submitted = st.form_submit_button("🔍 Predict Price", use_container_width=True)

# Make prediction
if submitted:
    # Prepare input data
    petrol = 1 if fuel_type == "Petrol" else 0
    diesel = 1 if fuel_type == "Diesel" else 0
    electric = 1 if fuel_type == "Electric" else 0
    
    input_data = pd.DataFrame({
        'km_driven': [km_driven],
        'mileage': [mileage],
        'age': [age],
        'Petrol': [petrol],
        'Diesel': [diesel],
        'Electric': [electric]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display results
    with col2:
        st.subheader("💰 Prediction Result")
        
    col_res1, col_res2 = st.columns([1, 1])
    
    with col_res1:
        st.markdown(f"""
            <div class="prediction-box">
                <h3 style="margin: 0; font-size: 1.2rem; opacity: 0.9;">Predicted Price</h3>
                <h1 style="margin: 0.5rem 0; font-size: 2.5rem;">₹ {prediction:.2f} Lakhs</h1>
                <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;">Based on your car details</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_res2:
        st.markdown("### 📊 Input Summary")
        summary_data = {
            "KM Driven": f"{km_driven:,}",
            "Mileage": f"{mileage} km/l",
            "Age": f"{age} years",
            "Fuel Type": fuel_type
        }
        for key, value in summary_data.items():
            st.markdown(f"**{key}:** {value}")

# Information section
st.markdown("---")
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("""
    <div class="metric-card">
        <h3>✅ Accurate Model</h3>
        <p>Trained on extensive car price data using XGBoost algorithm</p>
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    st.markdown("""
    <div class="metric-card">
        <h3>⚡ Instant Results</h3>
        <p>Get predictions in real-time with just a few inputs</p>
    </div>
    """, unsafe_allow_html=True)

with col_info3:
    st.markdown("""
    <div class="metric-card">
        <h3>🎯 Easy to Use</h3>
        <p>Simple and intuitive interface for everyone</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: gray; font-size: 0.85rem;">
    💡 <strong>Tip:</strong> The model predicts car prices based on kilometers driven, mileage, age, and fuel type. 
    Results are estimates and may vary from actual market prices.
</p>
""", unsafe_allow_html=True)
