import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------
# Page Configuration
# ------------------------

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide",
    page_icon="📊"
)

# ------------------------
# Styling
# ------------------------

st.markdown("""
<style>
.main {
    background-color:#f4f6f9;
}
.stButton>button {
    background-color:#1f77b4;
    color:white;
    border-radius:8px;
    height:3em;
    width:100%;
    font-weight:600;
}
.stButton>button:hover {
    background-color:#155a8a;
    color:white;
}
.result-box{
    padding:25px;
    border-radius:10px;
    background-color:white;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)


# ------------------------
# Load Model
# ------------------------

try:
    kmeans_model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    model_features = joblib.load("model_features.pkl")
    
    st.sidebar.success("✅ Models loaded successfully")
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.info("Please ensure kmeans_model.pkl, scaler.pkl, and model_features.pkl are in the same directory")
    st.stop()


# ------------------------
# Cluster Labels
# ------------------------

cluster_labels = {
    0: "Active_Bargain_Hunters",
    1: "Loyal_Web_Parents",
    2: "At_Risk_Customers",
    3: "Premium_Loyal_Customers"
}

segment_descriptions = {
    0: "Recent buyers who rely on discounts. Price-sensitive but engaged.",
    1: "Long-tenure customers with strong web engagement and consistent purchases.",
    2: "Inactive and discount-dependent customers. High churn risk.",
    3: "High-value loyal customers with strong purchasing power and low discount usage."
}


# ------------------------
# Title
# ------------------------

st.title("📊 Customer Segmentation Dashboard")
st.markdown("Predict a customer's behavioral segment based on purchasing patterns.")
st.markdown("---")


# ------------------------
# Session State Initialization
# ------------------------

defaults = {
    "recency": 0,
    "tenure": 0,
    "total_purchases": 0,
    "children": 0,
    "avg_order_value": 0.0,
    "numdealspurchases": 0,
    "numwebpurchases": 0,
    "deal_ratio": 0.0,
    "web_ratio": 0.0,
    "input_mode": "Raw Counts"  # Track input mode
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ------------------------
# Sidebar Dataset Upload
# ------------------------

st.sidebar.header("📂 Load Customer Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx"]
)

df_data = None

if uploaded_file:

    try:
        if uploaded_file.name.endswith(".csv"):
            df_data = pd.read_csv(uploaded_file)
        else:
            df_data = pd.read_excel(uploaded_file)

        st.sidebar.success("Dataset Loaded ✅")

        if st.sidebar.checkbox("Preview Data"):
            st.dataframe(df_data.head())

        # Show available columns for debugging
        with st.sidebar.expander("Available Columns"):
            st.write(list(df_data.columns))

        selected_index = st.sidebar.selectbox(
            "Select Customer Row",
            df_data.index
        )

        if st.sidebar.button("Load Selected Customer"):

            row = df_data.loc[selected_index]

            st.session_state.recency = int(row.get("Recency", 0))
            st.session_state.tenure = int(row.get("Customer_Tenure_Days", 0))
            st.session_state.total_purchases = int(row.get("Total_Purchases", 0))
            st.session_state.children = int(row.get("children", 0))
            st.session_state.avg_order_value = float(row.get("Avg_Order_Value", 0))
            
            # Load both raw counts and ratios if available
            st.session_state.numdealspurchases = int(row.get("NumDealsPurchases", 0))
            st.session_state.numwebpurchases = int(row.get("NumWebPurchases", 0))
            st.session_state.deal_ratio = float(row.get("Deal_Ratio", 0))
            st.session_state.web_ratio = float(row.get("Web_channel_ratio", 0))
            
            st.sidebar.success("Customer data loaded!")
            
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")


# ------------------------
# Input Form
# ------------------------

st.header("📝 Enter Customer Data")

# Input mode selector
input_mode = st.radio(
    "Select Input Method:",
    ["Raw Counts (NumDealsPurchases, NumWebPurchases)", 
     "Ratios (Deal_Ratio, Web_channel_ratio)"],
    horizontal=True,
    key="input_mode_selector"
)

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        recency = st.number_input(
            "Recency (Days since last purchase)",
            min_value=0,
            max_value=2000,
            value=st.session_state.recency,
            key="recency_input"
        )

        tenure = st.number_input(
            "Customer Tenure (Days with Company)",
            min_value=0,
            max_value=5000,
            value=st.session_state.tenure,
            key="tenure_input"
        )

        total_purchases = st.number_input(
            "Total Purchases",
            min_value=0,
            step=1,
            value=st.session_state.total_purchases,
            key="total_purchases_input"
        )

        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            step=1,
            value=st.session_state.children,
            key="children_input"
        )

    with col2:
        avg_order_value = st.number_input(
            "Average Order Value ($)",
            min_value=0.0,
            value=st.session_state.avg_order_value,
            key="avg_order_value_input"
        )

        # Conditional inputs based on mode
        if "Raw Counts" in input_mode:
            st.markdown("**📊 Enter Raw Counts:**")
            
            numdealspurchases = st.number_input(
                "Number of Deal Purchases",
                min_value=0,
                step=1,
                value=st.session_state.numdealspurchases,
                key="numdealspurchases_input",
                help="Will be converted to Deal_Ratio = Deal Purchases / Total Purchases"
            )

            numwebpurchases = st.number_input(
                "Number of Web Purchases", 
                min_value=0,
                step=1,
                value=st.session_state.numwebpurchases,
                key="numwebpurchases_input",
                help="Will be converted to Web_channel_ratio = Web Purchases / Total Purchases"
            )
            
            # Store values for later use
            st.session_state.numdealspurchases = numdealspurchases
            st.session_state.numwebpurchases = numwebpurchases
            
        else:  # Ratios mode
            st.markdown("**📈 Enter Ratios Directly:**")
            
            deal_ratio = st.slider(
                "Deal Ratio (0-1)",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.deal_ratio,
                step=0.01,
                help="Proportion of purchases made with deals (Deal Purchases / Total Purchases)"
            )
            
            web_ratio = st.slider(
                "Web Channel Ratio (0-1)",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.web_ratio,
                step=0.01,
                help="Proportion of purchases made through web (Web Purchases / Total Purchases)"
            )
            
            # Store values for later use
            st.session_state.deal_ratio = deal_ratio
            st.session_state.web_ratio = web_ratio

    predict_button = st.form_submit_button("🔍 Predict Customer Segment")
    
    # Update session state with form values
    if predict_button:
        st.session_state.recency = recency
        st.session_state.tenure = tenure
        st.session_state.total_purchases = total_purchases
        st.session_state.children = children
        st.session_state.avg_order_value = avg_order_value


# ------------------------
# Prediction Logic
# ------------------------

if predict_button:

    # Validation
    if st.session_state.total_purchases <= 0:
        st.error("Total Purchases must be greater than 0.")
        st.stop()

    # Calculate or validate ratios based on input mode
    if "Raw Counts" in input_mode:
        # Validate raw counts
        if st.session_state.numdealspurchases > st.session_state.total_purchases:
            st.error("Deal purchases cannot exceed total purchases.")
            st.stop()

        if st.session_state.numwebpurchases > st.session_state.total_purchases:
            st.error("Web purchases cannot exceed total purchases.")
            st.stop()
            
        # Calculate ratios from raw counts
        deal_ratio = st.session_state.numdealspurchases / st.session_state.total_purchases
        web_ratio = st.session_state.numwebpurchases / st.session_state.total_purchases
        
        # Store calculated ratios
        st.session_state.deal_ratio = deal_ratio
        st.session_state.web_ratio = web_ratio
        
    else:  # Ratios mode
        deal_ratio = st.session_state.deal_ratio
        web_ratio = st.session_state.web_ratio
        
        # Validate ratios (optional - can exceed 1 if both ratios sum > 1)
        if deal_ratio + web_ratio > 1.0:
            st.warning(f"⚠️ Deal Ratio ({deal_ratio:.2f}) + Web Ratio ({web_ratio:.2f}) = {deal_ratio + web_ratio:.2f} > 1. This means some purchases might be counted in both categories.")
        
        # For display purposes, calculate equivalent raw counts (if total_purchases > 0)
        if st.session_state.total_purchases > 0:
            st.session_state.numdealspurchases = int(deal_ratio * st.session_state.total_purchases)
            st.session_state.numwebpurchases = int(web_ratio * st.session_state.total_purchases)

    # Feature Engineering
    log_avg_value = np.log1p(st.session_state.avg_order_value)

    # Create DataFrame with correct feature names
    input_df = pd.DataFrame([[
        st.session_state.recency,
        st.session_state.tenure,
        st.session_state.total_purchases,
        log_avg_value,
        deal_ratio,  # Deal_Ratio
        web_ratio,   # Web_channel_ratio
        st.session_state.children
    ]], columns=model_features)

    # Debug info (optional)
    with st.expander("🔍 Debug Information - See What's Being Sent to Model"):
        st.write("**Input Features Sent to Model:**")
        debug_df = input_df.copy()
        debug_df.columns = ['Recency', 'Tenure', 'Total_Purchases', 'Log_Avg_Value', 
                           'Deal_Ratio', 'Web_Ratio', 'Children']
        st.dataframe(debug_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Raw Input Values:**")
            st.json({
                "Input Mode": input_mode,
                "Recency": st.session_state.recency,
                "Tenure": st.session_state.tenure,
                "Total Purchases": st.session_state.total_purchases,
                "Avg Order Value": f"${st.session_state.avg_order_value:.2f}",
                "Children": st.session_state.children
            })
        
        with col2:
            st.write("**Calculated/Input Ratios:**")
            st.json({
                "Deal Ratio": f"{deal_ratio:.3f}",
                "Web Ratio": f"{web_ratio:.3f}",
                "Raw Deal Purchases": st.session_state.numdealspurchases,
                "Raw Web Purchases": st.session_state.numwebpurchases
            })

    try:
        # Scale and predict
        scaled = scaler.transform(input_df)
        cluster = int(kmeans_model.predict(scaled)[0])

        segment = cluster_labels.get(cluster, f"Unknown Cluster {cluster}")
        description = segment_descriptions.get(cluster, "No description available")

        # Display results
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)

        st.subheader("🎯 Predicted Segment")
        
        # Color-coded clusters
        colors = ["#28a745", "#17a2b8", "#ffc107", "#dc3545"]
        st.markdown(f"<h2 style='color: {colors[cluster]}'>{segment} (Cluster {cluster})</h2>", 
                   unsafe_allow_html=True)

        st.write(description)
        
        # Additional metrics
        st.markdown("---")
        st.subheader("📊 Customer Profile Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Recency", f"{st.session_state.recency} days")
            st.metric("Tenure", f"{st.session_state.tenure} days")
        with col2:
            st.metric("Total Purchases", st.session_state.total_purchases)
            st.metric("Avg Order Value", f"${st.session_state.avg_order_value:.2f}")
        with col3:
            st.metric("Deal Ratio", f"{deal_ratio:.1%}")
            st.metric("Web Ratio", f"{web_ratio:.1%}")
            
        # Show both raw counts and ratios for clarity
        st.markdown("---")
        st.subheader("📈 Purchase Breakdown")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Raw Deal Purchases", st.session_state.numdealspurchases)
        with col2:
            st.metric("Raw Web Purchases", st.session_state.numwebpurchases)
        with col3:
            st.metric("Other Purchases", 
                     st.session_state.total_purchases - 
                     st.session_state.numdealspurchases - 
                     st.session_state.numwebpurchases)
        with col4:
            st.metric("Total", st.session_state.total_purchases)

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Please check that the model features match the input features")

# Add footer with instructions
st.markdown("---")
st.markdown("""
### 📌 Instructions
1. **Choose Input Method**: Select either Raw Counts or Ratios
2. **Enter Data**: Fill in the customer information
3. **Predict**: Click to see the customer segment

**Note**: 
- In **Raw Counts** mode: Enter actual purchase numbers (ratios calculated automatically)
- In **Ratios** mode: Enter proportions directly (0-1 scale)
- Total Purchases must be > 0 for ratio calculations
""")