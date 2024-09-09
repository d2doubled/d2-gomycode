import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model and encoders
model_path = r'C:\Users\HP\Desktop\Final_year_Project\stored_model.pkl'
encoders_path = r'C:\Users\HP\Desktop\Final_year_Project'

# Load the model
loaded_model = pickle.load(open(model_path, 'rb'))

# List of categorical features
categorical_features = [
    'Job Title', 'Company Name', 'Location', 'Type of ownership', 
    'Industry', 'Sector', 'Revenue', 'Competitors'
]

# Dictionary to store encoders
encoders = {}

# Load each encoder
for feature in categorical_features:
    encoder_file = f"stored_encoder_{feature}.pkl"
    encoders[feature] = pickle.load(open(encoders_path + '\\' + encoder_file, 'rb'))

# Define the correct order of columns
correct_column_order = [
    'Job Title', 'Rating', 'Company Name', 'Location', 'Type of ownership', 
    'Industry', 'Sector', 'Revenue', 'Competitors', 
    'python_yn', 'R_yn', 'spark', 'aws', 'excel', 'java', 'c++'
]

# Function to preprocess user inputs
def preprocess_input(job_title, rating, company_name, location, type_of_ownership, industry, sector,
                     revenue, competitors, python_yn, R_yn, spark, aws, excel, java, c_plus_plus):
    # Create a DataFrame from user inputs
    input_data = {
        'Job Title': [job_title],
        'Rating': [rating],
        'Company Name': [company_name],
        'Location': [location],
        'Type of ownership': [type_of_ownership],
        'Industry': [industry],
        'Sector': [sector],
        'Revenue': [revenue],
        'Competitors': [competitors],        
        'python_yn': [python_yn],
        'R_yn': [R_yn],
        'spark': [spark],
        'aws': [aws],
        'excel': [excel],
        'java': [java],
        'c++': [c_plus_plus]
    }
    
    input_df = pd.DataFrame(input_data)

    # Encode categorical features
    for feature in categorical_features:
        if feature in encoders:
            input_df[feature] = encoders[feature].transform(input_df[feature])

    # Reorder the columns
    input_df = input_df[correct_column_order]

    return input_df

# Streamlit app
st.title("Salary Prediction App")

# Sidebar for user inputs
st.sidebar.header("Input Features")

# Sidebar inputs using encoder classes
job_title = st.sidebar.selectbox("Job Title", encoders['Job Title'].classes_)
company_name = st.sidebar.selectbox("Company Name", encoders['Company Name'].classes_)
location = st.sidebar.selectbox("Location", encoders['Location'].classes_)
type_of_ownership = st.sidebar.selectbox("Type of Ownership", encoders['Type of ownership'].classes_)
industry = st.sidebar.selectbox("Industry", encoders['Industry'].classes_)
sector = st.sidebar.selectbox("Sector", encoders['Sector'].classes_)
revenue = st.sidebar.selectbox("Revenue", encoders['Revenue'].classes_)
competitors = st.sidebar.selectbox("Competitors", encoders['Competitors'].classes_)
rating = st.sidebar.slider("Company Rating", 0.0, 5.0, 4.0)
python_yn = st.sidebar.selectbox("Knowledge of Python", ["Yes", "No"])
R_yn = st.sidebar.selectbox("Knowledge of R", ["Yes", "No"])
spark = st.sidebar.selectbox("Knowledge of Spark", ["Yes", "No"])
aws = st.sidebar.selectbox("Knowledge of AWS", ["Yes", "No"])
excel = st.sidebar.selectbox("Knowledge of Excel", ["Yes", "No"])
java = st.sidebar.selectbox("Knowledge of Java", ["Yes", "No"])
c_plus_plus = st.sidebar.selectbox("Knowledge of C++", ["Yes", "No"])

# Convert Yes/No to 1/0
def convert_to_binary(value):
    return 1 if value == "Yes" else 0

python_yn = convert_to_binary(python_yn)
R_yn = convert_to_binary(R_yn)
spark = convert_to_binary(spark)
aws = convert_to_binary(aws)
excel = convert_to_binary(excel)
java = convert_to_binary(java)
c_plus_plus = convert_to_binary(c_plus_plus)

# Footer
st.markdown("## Instructions")
st.markdown("Use the sidebar to input features for the salary prediction. The app will output the predicted salary based on the inputs.")

# Predict button
if st.button("Predict Salary"):
    # Preprocess input and make prediction
    input_df = preprocess_input(job_title, rating, company_name, location, type_of_ownership,
                                industry, sector, revenue, competitors, 
                                python_yn, R_yn, spark, aws, excel, java, c_plus_plus)
    
    prediction = loaded_model.predict(input_df)
    st.write(f"Predicted Salary: ${np.round(prediction[0], 2)}")

