import streamlit as st
import pandas as pd
import pickle
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def load_model(filename):
    # "rb" read in file as binary
    with open(filename, "rb") as file:
        return pickle.load(file)
    
xgboost_model = load_model('CSV_PKL_Files/xgb_model.pkl')

naive_bayes_model = load_model("CSV_PKL_Files/nb_model.pkl")

random_forest_model = load_model("CSV_PKL_Files/rf_model.pkl")

decision_tree_model = load_model("CSV_PKL_Files/dt_model.pkl")

svm_model = load_model("CSV_PKL_Files/svm_model.pkl")

k_model = load_model("CSV_PKL_Files/k_model.pkl")

voting_classifier_model = load_model("CSV_PKL_Files/voting_clf.pkl")

xgboost_SMOTE_model = load_model("CSV_PKL_Files/xgboost-SMOTE.pkl")

xgboost_featureEngineered_model = load_model("CSV_PKL_Files/xgboost_model.pkl")

def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):
    input_dict = {
        'Credit Score': credit_score,
        'Age' : age,
        'Tenure' : tenure,
        'Balance' : balance,
        'NumOfProducts' : num_products,
        'HasCrCard' : has_credit_card,
        'IsActiveMember' : is_active_member,
        'EstimatedSalary' : estimated_salary,
        'Geography_France' : 1 if location == "France" else 0,
        'Geography_Germany' : 1 if location == "Germanuy" else 0,
        'Geography_Spain' : 1 if location == "Spain" else 0,
        'Gender_Male' : 1 if gender == "Male" else 0,
        'Gender_Female' : 1 if location == "Female" else 0
    }
    
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

def make_predictions(input_df, input_dict):
    probabilities = {
        'XGBoost' : xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest' : random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbors' : k_model.predict_proba(input_df)[0][1],
        # 'Naive Bayes' : naive_bayes_model.predict_proba(input_df)[0][1], (has a probability of 0)
        # 'Decision Tree' : decision_tree_model.predict_proba(input_df)[0][1], (has a probability of 0)
        # 'XGBoost Feature Engineered' : xgboost_featureEngineered_model.predict_proba(input_df)[0][1], (has other features we haven't defined yet)
        # 'XGBoost Smote' : xgboost_SMOTE_model.predict_proba(input_df)[0][1] (has other features we haven't defined yet)
        # 'Support Vector Machine' : svm_model.predict_proba(input_df)[0][1] (doesn't have a predict_proba function)
        # 'Vating Classifier' : voting_classifier_model.predict_proba(input_df)[0][1] (doesn't have a predict_proba function)
    }
    
    avg_probability = np.mean(list(probabilities.values()))
    
    st.markdown("### Model Probabilies")
    for model, proba in probabilities.items():
        st.write(f"{model} {proba}")
    st.write(f"Average Probability: {avg_probability}")

st.title("Customer Churn Prediction")

df = pd.read_csv("CSV_PKL_Files/churn.csv")
# print("---------------------")
# print(df)

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    #splitting the "ID - Name" by the dash
    selected_customerId, selected_customer_surname = selected_customer_option.split(" - ")
    #convert selected_customerId from a string to an int
    selected_customerId = int(selected_customerId)
    # print("Selected Customer ID: ", selected_customerId)
    # print("Selected Customer Surname: ", selected_customer_surname)

    #filter the dataframe to grab all data that is related to the selected customer
    selected_customer = df.loc[df['CustomerId'] == selected_customerId].iloc[0]
    
    print("Selected Customer Information:\n", selected_customer)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # UI component from Streamlit library that is a number input
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=int(selected_customer['CreditScore'])
        )
        
        # UI component from Streamlit library that is a dropdown
        location = st.selectbox(
            "Location", ["Spain", "France", "Germany"],
            index=["Spain", "France", "Germany"].index(
                selected_customer['Geography']
            )
        )
        
        # UI component from Streamlit library that is a 0 or 1 option
        gender = st.radio("Gender", 
                          ["Male", "Female"],
                          index=0 if selected_customer['Gender'] == 'Male' else 1
        )
        
        # UI component from Streamlit library that is a number input
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=int(selected_customer['Age'])
        )
        
        # UI component from Streamlit library that is a number input
        tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=50,
            value=int(selected_customer['Tenure'])
        )
        
    with col2:
        
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=float(selected_customer['Balance'])
        )
        
        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=10,
            value=int(selected_customer['NumOfProducts'])
        )
        
        has_credit_card = st.checkbox(
            "Has Credit Card",
            value=bool(selected_customer['HasCrCard'])
        )
        
        is_active_member = st.checkbox(
            "Is Active Member",
            value = bool(selected_customer['IsActiveMember'])
        )
        
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer['EstimatedSalary'])
        )
    
    # adding machine learning model and the predicitions the model is giving us
    input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)
    make_predictions(input_df, input_dict)
    