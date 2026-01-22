# Environments
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import plotly.express as px
import base64

# For Regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# For Classification
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Make graphs look nice
plt.style.use('seaborn-v0_8-whitegrid')

# Page config
st.set_page_config(page_title="Loans Analysis", page_icon=":dollar:", layout="wide")

# images
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

banner_b64 = get_base64_image("Banner6.jpg")
logo_b64 = get_base64_image("money_hypermarket2.png")

st.markdown(
    f"""
    <style>
    .image-container {{
        position: relative;
        width: 100%;
    }}

    .banner-img {{
        width: 100%;
        display: block;
    }}

    .logo-img {{
        position: absolute;
        bottom: 00%;       
        left: 0%;       
        width: 200px;   
        z-index: 10;    
    }}
    </style>

    <div class="image-container">
        <img src="data:image/jpg;base64,{banner_b64}" class="banner-img">
        <img src="data:image/png;base64,{logo_b64}" class="logo-img">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <h1 style="color:#000080; font-weight:bold;">
        Loan Industry Analysis
    </h1>
    <p style="color:#000080; font-weight:bold; font-size:1.1rem;">
        Global data analysis evaluating 50,000 previous customers and the key factors impacting loan approvals, interest rates and loan defaults
    </p>
    """,
    unsafe_allow_html=True
)
# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Loan_approval_data_2025.csv")
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Apply filters to explore loan profiles")

# Custom styling
page_style = """
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #121E50 !important;
}

/* Sidebar inner content */
[data-testid="stSidebar"] > div:first-child {
    background-color: #121E50 !important;
}

/* Sidebar text color */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main page background */
[data-testid="stAppViewContainer"] {
    background-color: #E6F0FF !important;  /* Pale Sky Blue */
}

</style>

"""
st.markdown(page_style, unsafe_allow_html=True)

def get_gif_base64(file_path):
    with open(file_path, "rb") as f:
        content = f.read()
        return base64.b64encode(content).decode("utf-8")

# Path to your local GIF
gif_path = 'penguin.gif'
gif_data = get_gif_base64(gif_path)

# Credit score filter
score_range = st.sidebar.slider(
    "Credit Score",
    min_value=int(df['credit_score'].min()),
    max_value=int(df['credit_score'].max()),
    value=(200, 800)
)

# Loan amount filter
loan_range = st.sidebar.slider(
    "Loan Amount",
    min_value=int(df['loan_amount'].min()),
    max_value=int(df['loan_amount'].max()),
    value=(0, 50000)
)

# Duration filter
duration_range = st.sidebar.slider(
    "Credit History (years)",
    min_value=int(df['credit_history_years'].min()),
    max_value=int(df['credit_history_years'].max()),
    value=(1, 10)
)

# Infidelity filter
delinquency_filter = st.sidebar.radio(
    "Delinquency Status",
    options=['All', 'No Delinquency', 'Delinquency Occurred']
)

# Employment status filter
employment_status__filter = st.sidebar.radio(
    "Employment Status",
    options=['All','Employed', 'Self-Employed', 'Student']
)

# Loan purpose filter
loan_purpose__filter = st.sidebar.radio(
    "Loan Purpose",
    options=['All','Business', 'Debt Consolidation', 'Education', 'Home Improvement', 'Medical', 'Personal']
)

# Loan type filter
all_loan_types = sorted(df['product_type'].dropna().unique())
loan_type_filter = st.sidebar.multiselect(
    "Loan Type",
    options=all_loan_types,
    default=all_loan_types   # e.g. ["Credit Card"] in your case
)
# Apply filters
filtered_df = df[
    (df['credit_score'] >= score_range[0]) & 
    (df['credit_score'] <= score_range[1]) &
    (df['credit_history_years'] >= duration_range[0]) & 
    (df['credit_history_years'] <= duration_range[1]) &
    (df['loan_amount'] >= loan_range[0]) & 
    (df['loan_amount'] <= loan_range[1]) &
    (df['product_type'].isin(loan_type_filter))
]

if delinquency_filter == 'No Delinquency':
    filtered_df = filtered_df[filtered_df['delinquencies_last_2yrs'] == 0]
elif delinquency_filter == 'Delinquency Occurred':
    filtered_df = filtered_df[filtered_df['delinquencies_last_2yrs'] == 1]

if employment_status__filter == 'Employed':
    filtered_df = filtered_df[filtered_df['occupation_status'] == "Employed"]
elif employment_status__filter == 'Self-Employed':
    filtered_df = filtered_df[filtered_df['occupation_status'] == "Self-Employed"]
elif employment_status__filter == 'Student':
    filtered_df = filtered_df[filtered_df['occupation_status'] == "Student"]

if loan_purpose__filter == 'Business':
    filtered_df = filtered_df[filtered_df['loan_intent'] == "Business"]
elif loan_purpose__filter == 'Debt Consolidation':
    filtered_df = filtered_df[filtered_df['loan_intent'] == "Debt Consolidation"]
elif loan_purpose__filter == 'Education':
    filtered_df = filtered_df[filtered_df['loan_intent'] == "Education"]
elif loan_purpose__filter == 'Home Improvement':
    filtered_df = filtered_df[filtered_df['loan_intent'] == "Home Improvement"]
elif loan_purpose__filter == 'Medical':
    filtered_df = filtered_df[filtered_df['loan_intent'] == "Medical"]
elif loan_purpose__filter == 'Personal':
    filtered_df = filtered_df[filtered_df['loan_intent'] == "Personal"]

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dataset Overview", "Customer Analysis", "Loan Approval Prediction", "Interest Rate Prediction", "Lender Risk Analysis"])

with tab1:
    # Basic info
    st.header("Dataset Overview - Matching Filtered Profiles")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Customers With Filtered Profile", len(filtered_df))
    with col2:
        st.metric("Loan Approval Rate", f"{filtered_df['loan_status'].mean():.1%}")
    with col3:
        st.metric("Average Interest Rate", f"{filtered_df['interest_rate'].mean()/100:.1%}")
    with col4:
        pct_shown = len(filtered_df) / len(df) * 100
        st.metric("Total Data Shown", f"{pct_shown:.1f}%")

    st.subheader("Filtered Data Sample")
    st.dataframe(df.head())

with tab2:
    st.header("Historic Loan Applications - Filtered Profiles")
    
    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        # Credit score distribution
        filtered_df["loan_status_label"] = filtered_df["loan_status"].map({0: "Loan not approved", 1: "Loan approved"})
        st.subheader("Credit Score vs Loan Approval")
        fig2, ax2 = plt.subplots()
        sns.histplot(data=filtered_df, x="credit_score", bins=30, hue="loan_status_label", multiple="stack", edgecolor="black", ax=ax2)
        ax2.set_xlabel("Credit Score")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)
        st.markdown("The above chart displays the number of approved loans versus not approved loans within your selected credit score range."
        "\n\nAs you can see the distribution of **loan approvals** is skewed heavily in favour of **higher credit scores**."
        "\n\nYour credit score has the **highest impact** on your chances of securing a loan."
        "\n\nCheck what can be done to improve your credit score via your online banking **mobile app**.")

    with col2:
        # Approval Distribution
        st.subheader("Loan Approval Distribution")
        fig1, ax1 = plt.subplots()
        approval_counts = filtered_df["loan_status"].value_counts()
        colors = ["#e74c3c", "#4A69BD"]
        ax1.pie(
            approval_counts,
            labels=["Not Approved", "Approved"],
            autopct="%1.1f%%",
            colors=colors,
        )
        st.pyplot(fig1)

    # Calculate percentage of approved loans
    total_loans = len(filtered_df)
    approved_loans = len(filtered_df[filtered_df['loan_status']==1])
    loans_approved_pct = (approved_loans / total_loans) * 100

    st.info(f"#### The number of approved loans in your filtered selection is: **{loans_approved_pct:.1f}%**")
    
    st.markdown(
        """
        <hr style="border: 1.5px solid #444; margin: 1rem 0;" />
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <h3 style="text-align:center; font-size:1.6rem;">
            Average Interest Rates by Loan Type For Customers
        </h3>
        """,
        unsafe_allow_html=True
    )

    interest_on_loans = filtered_df['interest_rate'].mean()
    st.info(f"#### The avereage interest rate in your filtered selection is: **{interest_on_loans:.1f}%**")

    st.markdown(
    """
    - The below heatmap plots the loan type against interest rates to illustrate the optimum loan selection for your purposes
    - The **cheapest interest rates** are available when applying for a **line of credit** 
    - The most **expensive interest rates** are linked to **credit cards** although credit cards are often the quickest and simplest way to access credit. 
    - Loan purpose has an impact on your interest rate, when applying for a loan please ensure you select the most suitable:

        - **loan type,** 
        - **loan purpose**, and;
        - **employment status**.
    """
    )
    pivot_data = filtered_df.loc[filtered_df['loan_status'] == 1, 
                            ['loan_intent', 
                            'product_type', 
                            'occupation_status', 
                            'loan_status', 
                            'loan_amount', 
                            'interest_rate']]

    df_pivot = pd.pivot_table(data=pivot_data, 
                            index=['loan_intent', 'product_type'],
                            columns=['occupation_status'],
                            values=['interest_rate'],
                            aggfunc={'interest_rate': 'mean'})

    df_pivot = df_pivot.round(2)

    fig1, ax = plt.subplots(figsize=(10,7))

    sns.heatmap(data=df_pivot, 
                annot=True, 
                linewidths=.2, 
                cmap='icefire', 
                fmt=".2f", 
                ax=ax)

    ax.xaxis.tick_top()  # Move x-axis labels to the top
    ax.set_title("", fontsize= 17)
    fig1.tight_layout()
    st.pyplot(fig1)

# Machine Learning Model
with tab3:
    st.header("Machine Learning - Random Forest Model")
    st.write("We trained a machine learning model on unique 50,000 data points which can predict the outcome of your loan application for FREE!")

    # Prepare data with selected features only
    @st.cache_data
    def prepare_data(dataframe):

        # --- Select ONLY the specified features ---
        selected_features = [
            'credit_score',
            'loan_amount',
            'credit_history_years',
            'occupation_status',
            'loan_intent',
            'product_type',
            'age',
            'debt_to_income_ratio',
            'delinquencies_last_2yrs',
            'annual_income'
        ]

        # Independent variables
        X = dataframe[selected_features].copy()

        # Target variable
        y = dataframe['loan_status'].copy()

        # --- Encode categorical variables ---
        label_encoders = {}
        categorical_cols = X.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

        return X, y, label_encoders
    
    # Train model button
    if st.button("Train Model"):
        with st.spinner("Training Random Forest..."):
            # Prepare data
            X, y, encoders = prepare_data(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Store in session state
            st.session_state['model'] = model
            st.session_state['encoders'] = encoders
            st.session_state['features'] = X.columns.tolist()
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Performance")
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{accuracy:.2%}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(10)
                
                fig2, ax2 = plt.subplots()
                ax2.barh(importance_df['feature'], importance_df['importance'])
                ax2.set_xlabel('Importance')
                st.pyplot(fig2)
            
            st.success("Model trained successfully!")
    
    # Display existing model info
    if 'model' in st.session_state:
        st.info("Model is ready for predictions!")
        
        # Show top features
        st.subheader("Top 5 Crucial Factors in Model Predictions")
        st.write("These elements will most heavily impact your chances of loan approval:")
        model = st.session_state['model']
        features = st.session_state['features']
        
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(5)
        
        for idx, row in importance_df.iterrows():
            st.write(f"- **{row['feature']}**")

    # Prediction Interface
    st.markdown("---")
    st.header("Make a Prediction about your Loan")
        
    if 'model' in st.session_state:
        st.write("Enter the details below to predict loan approval probability:")
            
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
                
            with col1:
                credit_score_input = st.number_input(
                    "Credit Score",
                    min_value=0,
                    max_value=900,
                    value=25,
                    step=10
                )
                    
                loan_amount__input = st.number_input(
                    "Loan Amount",
                    min_value=0,
                    max_value=200000,
                    value=10000,
                    step=1000
                )
                    
                credit_history_input = st.number_input(
                    "Credit History Years",
                    min_value=0,
                    max_value=20,
                    value=0,
                    step=1
                )

                income_input = st.slider(
                    "Income",
                    min_value=15000,
                    max_value=200000,
                    value=50000,
                    step=1000
                )
            with col2:
                occupation_input = st.selectbox(
                    "Occupation Status",
                    options=[0, 1, 2],
                    format_func=lambda x: {0: "Employed", 1: "Self Employed", 2: "Student"}[x]
                )
                            
                # Add more inputs based on your dataset features
                # Example placeholders - replace with actual features from your CSV:
                loan_intent_input = st.selectbox(
                    "Loan Purpose",
                    options=[0, 1, 2, 3, 4, 5],
                    format_func=lambda x: {0: "Business", 1: "Debt Consolidation", 2: "Education", 3: "Home Emprovement", 4: "Medical", 5: "Personal"}[x]
                )
                    
                loan_type_input = st.selectbox(
                    "Loan Type",
                    options=[0, 1, 2],
                    format_func=lambda x: {0: "Credit Card", 1: "Line of Credit", 2: "Personal Loan"}[x]
                )
                
            with col3:
                # Add more feature inputs as needed
                age_input = st.slider(
                    "Age",
                    min_value=15,
                    max_value=80,
                    value=25
                )
                    
                debt_ratio_input = st.slider(
                    "Current Debt Ratio",
                    min_value=0.0,
                    max_value=0.9,
                    value=0.5,
                    step=0.1
                )
                    
                missed_payment_input = st.selectbox(
                    "Missed Payments Last 2 Years",
                    options=[0, 1 ],
                    format_func=lambda x: {0: "No", 1: "Yes"}[x]
                )

            # Submit button
            submit_button = st.form_submit_button("Predict Loan Approval Probability")
            
            if submit_button:
                # Create input dataframe with all features from ML model
                # Note: You'll need to match the exact feature names from your dataset
                input_data = pd.DataFrame({
                    'credit_score': [credit_score_input],
                    'loan_amount': [loan_amount__input],
                    'credit_history_years': [credit_history_input],
                    'occupation_status': [occupation_input],
                    'loan_intent': [loan_intent_input],
                    'product_type': [loan_type_input],
                    'age': [age_input],
                    'debt_to_income_ratio': [debt_ratio_input],
                    'delinquencies_last_2yrs': [missed_payment_input],
                    'annual_income': [income_input]
                })
                
                # Ensure all features from training are present
                # Add any missing features with default values if needed
                for feature in st.session_state['features']:
                    if feature not in input_data.columns:
                        input_data[feature] = 0  # or appropriate default
                
                # Reorder columns to match training data
                input_data = input_data[st.session_state['features']]
                
                # Encode categorical variables if any
                for col in input_data.columns:
                    if col in st.session_state['encoders']:
                        # Handle encoding for categorical features
                        pass  # Implement if you have categorical features
                
                # Make prediction
                prediction = st.session_state['model'].predict(input_data)[0]
                prediction_proba = st.session_state['model'].predict_proba(input_data)[0]
                
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction > 0.7:
                        st.error("**High Chance of Loan Approval**")
                        st.metric("Loan Approval Probability", f"{prediction_proba[1]:.1%}")
                    else:
                        st.success("**Low Chance of Loan Approval**")
                        st.metric("Loan Approval Probability", f"{prediction_proba[0]:.1%}")
                
                with col2:
                    st.write("**Probability Breakdown:**")
                    st.write(f"- Loan Approved: {prediction_proba[0]:.1%}")
                    st.write(f"- Loan Not Approved: {prediction_proba[1]:.1%}")
                    
                    # Visual representation
                    fig, ax = plt.subplots(figsize=(6, 2))
                    categories = ['Approved', 'Not Approved']
                    probabilities = prediction_proba
                    colors = ["#FF8C00", "#121E50"]
                    ax.barh(categories, probabilities, color=colors)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probability')
                    for i, v in enumerate(probabilities):
                        ax.text(v + 0.02, i, f'{v:.1%}', va='center')
                    st.pyplot(fig)

    else:
        st.warning("Please train the model first using the 'Train Model' button above.")


with tab4:
    st.header("Linear Regression Model")
    
    # Prepare data
    @st.cache_data
    def prepare_data(dataframe):
        # Copy data
        X = dataframe.drop('interest_rate', axis=1).copy()
        y = dataframe['interest_rate'].copy()
        
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        
        return X, y, label_encoders
    
    # Train model button
    if st.button("Train Interest Rate Predictor Model"):
        with st.spinner("Training Linear Regression..."):
            # Prepare data
            X2, y2, encoders2 = prepare_data(df)
            
            # Split data
            X_train2, X_test2, y_train2, y_test2 = train_test_split(
                X2, y2, test_size=0.2, random_state=42,
            )
            
            # Train model
            model2 = LinearRegression()
            model2.fit(X_train2, y_train2)
            
            # Predictions
            y_pred2 = model2.predict(X_test2)
            
            # Store in session state
            st.session_state['model2'] = model2
            st.session_state['encoders2'] = encoders2
            st.session_state['features2'] = X2.columns.tolist()
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:

                mae= mean_absolute_error(y_test2, y_pred2)
                rmse = np.sqrt(mean_squared_error(y_test2, y_pred2))
                r2 = r2_score(y_test2, y_pred2)
                
                # Put metrics in a small DataFrame
                metrics_df = pd.DataFrame(
                    {
                        "Metric": ["R² (Explained Variance)", "Mean Absolute Error", "Root Mean Squared Error"],
                         "Value": [r2, mae, rmse],
                    }
                )

                # Style it a bit
                styled_metrics = (
                    metrics_df.style
                    .format({"Value": "{:.3f}"})
                    .hide(axis="index")
                    .set_table_styles(
                        [
                            # Header
                            {
                                "selector": "th.col_heading",
                                "props": "background-color:#0047AB; color:white; font-weight:bold; text-align:center;"},
                            {"selector": "th.index_name", "props": "display:none;"},
                            # Whole table background
                            {
                                "selector": "tbody td",
                                "props": "background-color:#E6F2FF; text-align:center; padding:6px;"
                            },
                            {
                                "selector": "tbody tr:nth-child(even) td",
                                "props": "background-color:#D6EAF8;"
                            },
                        ]
                    )
                )
            st.success("Model trained successfully!")
            st.subheader("Predictive Model Performance")
            st.dataframe(styled_metrics, use_container_width=True)

            # Take first 200 points for line comparison clarity
            y_test_line = y_test2.iloc[:200].reset_index(drop=True)
            y_pred_line = pd.Series(y_pred2[:200])

            # Line Graph (Actual vs Predicted)

            fig, ax = plt.subplots()
            ax.plot(y_test_line, label="Actual", color="green")
            ax.plot(y_pred_line, label="Predicted", color="red")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Interest Rate")
            ax.set_title("Actual vs Predicted Interest Rates")
            ax.legend()
            plt.tight_layout()

            st.pyplot(fig)

    # Display existing model info
    if 'model2' in st.session_state:
        st.info("Model is ready for predictions! Check the next script for prediction interface.")

with tab5:
    st.header("Target Markets and Default Risks")
    st.write("The below scatter graph plots customer defaults against credit rating and length of their credit history")

    # Ensure 'defaults_on_file' is categorical Yes/No
    df['defaults_on_file'] = df['defaults_on_file'].map({0: "No", 1: "Yes"})
    df['defaults_numeric'] = df['defaults_on_file'].map({"No": 0, "Yes": 1})

    # Calculate default rates
    upper_right = df[(df['credit_score'] > 600) & (df['credit_history_years'] > 15)]
    lower_left = df[(df['credit_score'] < 600) & (df['credit_history_years'] < 15)]

    default_rate_upper = upper_right['defaults_numeric'].dropna().mean() * 100
    default_rate_lower = lower_left['defaults_numeric'].dropna().mean() * 100

    # Create summary table
    summary_table = pd.DataFrame({
        "Credit Segment": [
            "Upper-Right Quadrant (Credit Score > 600 & Credit History > 15 yrs)",
            "Lower-Left Quadrant (Credit Score < 600 & Credit History < 15 yrs)"
        ],
        "Default Rate (%)": [
            f"{default_rate_upper:.1f}%",
            f"{default_rate_lower:.1f}%"
        ]
    })

    # Convert DataFrame to Markdown table
    table_md = summary_table.to_markdown(index=False)
        
    # Display in Streamlit
    st.markdown(table_md)
    st.write(f"The risk of default is **{default_rate_lower/default_rate_upper:.1f} times** more likely in the lower-left quadrant compared to the upper-right quadrant ")
    
    # Define custom colors
    color_map = {"Yes": "#FF8C00", "No": "#121E50"}

    fig4 = px.scatter(
        df,
        x='credit_history_years',
        y='credit_score',
        color='defaults_on_file',
        color_discrete_map=color_map,  # <-- Apply custom colors
        title="Credit Score vs Credit History"
    )
    # Add bright green dotted vertical line at x = 15
    fig4.add_vline(
    x=15,
    line_width=3,
    line_dash="dot",
    line_color="lime"   # bright green
)
    # Add bright green dotted horizontal line at y = 600
    fig4.add_hline(
    y=600,
    line_width=3,
    line_dash="dot",
    line_color="lime"   # bright green
)
    fig4.update_layout(
        title_x=0.3,
        title_font=dict(size=20)
    )
    # Display in Streamlit
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("##### We continue the analysis by showing employment status segments against credit scores to identify potential lower risk target audiences")

    st.markdown("---")

    fig = px.box(
        df,
        x='occupation_status',
        y='credit_score',
        color='occupation_status',
        points="all",  # show individual points
        title="Credit Score Distribution by Occupation"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("##### Surprisingly there is little to distinguish credit scores between employment statuses. Below we explore default rates by market segments.")
    st.markdown("---")

    st.header("Market Sizes and Market Values")
    st.write("The treemap below depicts the various market scales (loan amounts) versus market values (interest rates)")

    # Treemap
    fig5 = px.treemap(
        data_frame=df,  # <-- use df
        path=['occupation_status', 'loan_intent', 'product_type'],
        values='loan_amount',
        color='interest_rate',
        hover_name='credit_score',  # <-- column name, not f-string
        hover_data=['credit_score'],
        color_continuous_scale='plasma',
        title="Treemap by Occupation, Loan Intent & Product Type"
    )
    # Optionally adjust layout further
    fig5.update_layout(
        title_x=0.25  # center the title
    )

    # Display in Streamlit
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("""
                ##### The size of the boxes above represents market sizes and the colour shows interest rates
                - The largest market share is for employed people with credit cards and personal loans
                - Credit cards have the highest interest rates and represent the highest market value
                - In the next visuals we explore the associated default risks of various markets""")
    st.markdown("---")
    st.subheader("Aggregated Table to Sort and Assess Market Segments")

    # Convert Yes/No to numeric for calculation (temporary column)
    df['defaults_numeric'] = df['defaults_on_file'].map({'Yes': 1, 'No': 0})

    # Group the data INCLUDING occupation status
    market_share_table = (
        df.groupby(['product_type', 'loan_intent', 'occupation_status'])
        .agg(
            total_market_size=('loan_amount', 'sum'),
            avg_interest_rate=('interest_rate', 'mean'),
            avg_default_rate=('defaults_numeric', 'mean')
        )
        .reset_index()
    )

    # Convert default rate to percentage
    market_share_table['avg_default_rate'] = market_share_table['avg_default_rate'] * 100

    # Sort by market size descending
    market_share_table = market_share_table.sort_values(
        by='total_market_size',
        ascending=False
    ).reset_index(drop=True)

    # Add clean index starting at 1
    market_share_table.index = market_share_table.index + 1

    # Rename columns
    market_share_table = market_share_table.rename(columns={
        'product_type': 'Loan Type',
        'loan_intent': 'Loan Use',
        'occupation_status': 'Occupation Status',
        'total_market_size': 'Market Size ($)',
        'avg_interest_rate': 'Avg. Interest Rate (%)',
        'avg_default_rate': 'Avg. Default Rate (%)'
    })

    # Apply formatting
    styled_table = market_share_table.style.format({
        'Market Size ($)': "${:,.0f}",
        'Avg. Interest Rate (%)': "{:.1f}%",
        'Avg. Default Rate (%)': "{:.1f}%"
    })

    # Display in Streamlit
    st.dataframe(styled_table, use_container_width=True)
    st.markdown("##### Download the above data to your computer using the download button on the top right")
    st.markdown("---")

    # Sunburst: Size = Market Size, Color = Avg. Default Rate
    fig_sun = px.sunburst(
        market_share_table,
        path=['Loan Type', 'Occupation Status'],
        values='Market Size ($)',                   # wedge size
        color='Avg. Default Rate (%)',             # wedge color
        color_continuous_scale='Plasma',
        hover_data={
            'Market Size ($)': ':,.0f',
            'Avg. Interest Rate (%)': ':.1f',     # included in hover
            'Avg. Default Rate (%)': ':.1f'
        }
    )

    fig_sun.update_layout(
        title="Market Size by Loan Type & Occupation Status",
        title_x=0.24,
        margin=dict(t=60, l=10, r=10, b=10)
    )

    st.plotly_chart(fig_sun, use_container_width=True)
    st.markdown("""
                    ##### Market size is shown for each segment and the colour represents default rate
                    - Self-employed Market share is smaller than the Emplyed Segment, however;
                    - The Self-Employed default rate is considerably lower
                    - Consider targeting a marketing campaign for self-employed credit cards to optimise market value and defaults risks
                    ### Conclusion: Self employed credit cards offer an excellent market growth opportinuty for optimising market value, market scale and default risks
                """)
# Footer
st.markdown("---")
st.markdown("**©** Copyright protected by Money Hypermarché Inc. est. 2026 (time-travel is the surest way to be ahead of your time)")

# Row container with quote on the right
st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: space-between; width: 100%; margin-top: 20px;">
        <img src="data:image/gif;base64,{gif_data}" alt="Animated GIF" width="300">
        <div style="color: #001f3f; font-size: 32px; font-weight: bold; margin-left: 20px; text-align: right;">
            "Take the money and run<br>for the hills"<br>– Pengu
        </div>
    </div>
    """,
    unsafe_allow_html=True
)