import streamlit as st
import pandas as pd
import pickle
import catboost
import numpy

def predict_fin_inclusion(model, data_for_prediction):
    
    predictions_data = model.predict(data_for_prediction)
    predicted_proba = model.predict_proba(data_for_prediction).tolist()
    
    if predictions_data == 0:
    
    	st.subheader('The person does NOT own a bank account with a probability of: ' + str(round(predicted_proba[0][0]*100 , 3)) + ' %')
    else:
	    st.subheader('The person OWNS a bank account with a probability of: ' + str(round(predicted_proba[0][1]*100 , 3)) + ' %')
    
with open('CatBoostApp.pkl', 'rb') as file:  
    model = pickle.load(file)
    
st.set_page_config(layout="wide")    
st.image(image='Financial-inclusion.png')
st.title('Financial Inclusion classifier')
st.write('This is a web app to classify if a person owns a bank account based on \
         several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction')
        
country = st.sidebar.selectbox('Country', ('Rwanda','Tanzania','Kenya','Uganda'))
location_type = st.sidebar.selectbox('Location type',('Urban','Rural'))
cellphone_access = st.sidebar.selectbox('Cellphone access',('Yes','No'))
household_size = st.sidebar.slider(label = 'Household size', 
                                   min_value = 1,
                                   max_value = 21,
                                   value = 10,
                                   step = 1)
age_of_respondent = st.sidebar.slider(label = 'Age', 
                                   min_value = 16,
                                   max_value = 100,
                                   value = 50,
                                   step = 1)
gender_of_respondent = st.sidebar.selectbox('Gender',('Male','Female'))
relationship_with_head = st.sidebar.selectbox('Relationship with head',('Head of Household','Spouse','Child','Parent','Other relative'))
marital_status = st.sidebar.selectbox('Marital status',('Married/Living together','Single/Never Married','Widowed','Divorced/Seperated','Dont know'))
education_level = st.sidebar.selectbox('Education level',('Primary education','No formal education','Secondary education','Tertiary education','Vocational/Specialised training'))
job_type = st.sidebar.selectbox('Job type',('Self employed','Informally employed','Farming and Fishing','Remittance Dependent','Other Income','Formally employed Government','Government Dependent','Dont Know/Refuse to answer'))

features = {'country': country,'location_type': location_type,'cellphone_access': cellphone_access,'household_size': household_size,'age_of_respondent': age_of_respondent,'gender_of_respondent' : gender_of_respondent,'relationship_with_head' : relationship_with_head,'marital_status' : marital_status,'education_level': education_level,'job_type' : job_type}

df = pd.melt(pd.DataFrame([features]),value_vars=['country','location_type','cellphone_access','household_size','age_of_respondent','gender_of_respondent','relationship_with_head','marital_status','education_level','job_type'])

st.write(df)

if st.button('Predict'):
    
    predict_fin_inclusion(model, features_df)
