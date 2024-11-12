import streamlit as st
import pandas as pd
import pickle
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import utils as ut
import os

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key = os.getenv("GROQ_API_KEY")
)


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
        'CreditScore': credit_score,
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
        'Gender_Female' : 1 if location == "Female" else 0,
    }
    
    # SMOTE_input_dict = {
    #     'CreditScore': credit_score,
    #     'Age' : age,
    #     'Tenure' : tenure,
    #     'Balance' : balance,
    #     'NumOfProducts' : num_products,
    #     'HasCrCard' : has_credit_card,
    #     'IsActiveMember' : is_active_member,
    #     'EstimatedSalary' : estimated_salary,
    #     'Geography_France' : 1 if location == "France" else 0,
    #     'Geography_Germany' : 1 if location == "Germanuy" else 0,
    #     'Geography_Spain' : 1 if location == "Spain" else 0,
    #     'Gender_Male' : 1 if gender == "Male" else 0,
    #     'Gender_Female' : 1 if location == "Female" else 0,
    #     'AgeGroup_Elderly' : 1 if age >= 60 else 0,
    #     'AgeGroup_Senior' : 1 if age >= 45 and age < 60  else 0,
    #     'AgeGroup_MiddleAge' : 1 if age >= 30 and age < 45  else 0,
    #     'AgeGroup_Young' : 1 if age >= 0 and age < 30 else 0,
    #     'CLV' : (balance * estimated_salary) / 100000,
    #     'TenureAgeRatio' : tenure / age,
    #     'SMOTE': SMOTE(random_state=42)
    # }
    
    input_df = pd.DataFrame([input_dict])
    # input_SMOTE_df = pd.DataFrame([SMOTE_input_dict])
    return input_df, input_dict

def make_predictions(input_df, input_dict):
    probabilities = {
        'XGBoost' : xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest' : random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbors' : k_model.predict_proba(input_df)[0][1],
        # 'Naive Bayes' : naive_bayes_model.predict_proba(input_df)[0][1], (has a probability of 0)
        # 'Decision Tree' : decision_tree_model.predict_proba(input_df)[0][1], (has a probability of 0)
        # 'XGBoost Feature Engineered' : xgboost_featureEngineered_model.predict_proba(input_df)[0][1], # (has other features we haven't defined yet)
        # 'XGBoost Smote' : xgboost_SMOTE_model.predict_proba(input_SMOTE_df)[0][1] # (has other features we haven't defined yet)
        # 'Support Vector Machine' : svm_model.predict_proba(input_df)[0][1] (doesn't have a predict_proba function)
        # 'Vating Classifier' : voting_classifier_model.predict_proba(input_df)[0][1] (doesn't have a predict_proba function)
    }
    
    avg_probability = np.mean(list(probabilities.values()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The customer has a {avg_probability:.2%} probability of churning.")
        
    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)
    
    return avg_probability
    
    # st.markdown("### Model Probabilies")
    # for model, proba in probabilities.items():
    #     st.write(f"{model} {proba}")
    # st.write(f"Average Probability: {avg_probability}")
    
    # return avg_probability
    
def explain_prediction(probability, input_dict, surname):
    
    prompt = f"""You are ann expert data scientist at a bank, where you specialize in interpreting and explaining prediction of machine learning models.
    
    Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning, based on the
    information provided below.
    
    Here is the customer's information: {input_dict}
    
    Here are the machine learning model's top 10 most important features for predicting churn:
    
                  feature  |  importance
    0         CreditScore  |  0.035005
    1                 Age  |  0.109550
    2              Tenure  |  0.030054
    3             Balance  |  0.052786
    4       NumOfProducts  |  0.323888
    5           HasCrCard  |  0.031940
    6      IsActiveMember  |  0.164146
    7     EstimatedSalary  |  0.032655
    8    Geography_France  |  0.046463
    9   Geography_Germany  |  0.091373
    10    Geography_Spain  |  0.036855
    11      Gender_Female  |  0.045283
    12        Gender_Male  |  0.000000
    
    {pd.set_option('display.max_columns', None)}
    
    Here are the summary statistics for churned customers: {df[df['Exited'] == 1].describe()}
    
    Here are the summary statistics for non-churned customers: {df[df['Exited'] == 0].describe()}
    
    - If {round(probability * 100, 1)} is less than 40%, generate a 3 sentence explanation of why the customer might not be at risk of churning and use their information in the explanation and in the explanation and start the explanation with "They might not be at risk of churning because" and give the explanation.
    - If {round(probability * 100, 1)} is over 40%, generate a 3 sentence explanation of why the customer is at risk of churning and use their information in the explanation and start the explanation with "They are at risk of churning because" and give the explanation.
    - Your explanation should be based on the customer's information, the sumary statistics, of churned and non-churned customers, and the feature of imporatnces provided.
    - Use the customer's information when explaining things
    
    
    Don't mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model's prediction and top 10 most
    important features", or mention any of the customer's raw data, just explain the prediction.
    
    Don't mention any of the intructions given in the prompt.
    
    Only write an explanation for EITHER over a 40% risk of churning OR under a 40% risk of churning depending on {round(probability * 100, 1)}% 
    """
    
    print("EXPLANATION PROMPT", prompt)
    
    raw_response = client.chat.completions.create(
        model = "llama-3.2-11b-text-preview",
        messages=[{
            "role" : "user",
            "content" : prompt
        }],
    )
    return raw_response.choices[0].message.content

def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""You are a manager at CAP Bank. You are responsible for ensuring customers stay with the bank and are incentivezed with various offers.
    
    You noticed a customer named {surname} has a {round(probability * 100,1 )}% probability of churning. 
    
    Here is the customer's information: {input_dict}
    
    Here is some explanation as to why the customer might be at risk of churning: {explanation}
    
    Generate an email to the customer based on their information, asking them to stay if they are at risk of chruning, or offering them incentives so that they
    become more loyal to the bank.
    
    Make sure to list out a set of incentives to stay based on their information, in bullet point format. Don't ever mention the probability of churning, or
    the machine learning model to the customer.
    """
    
    raw_response = client.chat.completions.create(
        model="llama-3.2-11b-text-preview",
        messages=[{
            "role":"user",
            "content": prompt
        }],
    )
    
    print("\n\nEMAIL Prompt", prompt)
    
    return raw_response.choices[0].message.content

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
    
    # print(col1, col2)
    
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
        
        # age_group = st.radio("Age Group", 
        #                   ["Young", "Middle Age", "Senior", "Elderly"],
        #                   index=(
        #                       0 if selected_customer['AgeGroup'] == 'Young' else
        #                       1 if selected_customer['AgeGroup'] == 'Middle Age' else
        #                       2 if selected_customer['AgeGroup'] == 'Senior' else
        #                       3 if selected_customer['AgeGroup'] == 'Elderly' else
        #                       0  # Default to index 0 if none match
        #                 )
        # )
    
    # adding machine learning model and the predicitions the model is giving us
    input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)
    avg_probability= make_predictions(input_df, input_dict)
    
    explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])
    
    st.markdown("---")
    st.subheader("Explanation of Prediction")
    st.markdown(explanation)
    
    email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])
    
    st.markdown("---")
    st.subheader("Personalized Email")
    st.markdown(email)
    