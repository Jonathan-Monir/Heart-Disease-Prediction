import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier

import numpy as np
PreProcessed_df = pd.read_csv("PreProcessed_Heart_Disease.csv")

data = {
    'Column': ['Age', 'Gender', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR',
               'ST depression', 'Slope of ST', 'work_type', 'smoking_status', 'Heart Disease'],
    'Description': ['The person\'s age in years.', 'The person\'s gender.',
                    'The type of chest pain experienced by the person.', 'Blood Pressure in mmHg.',
                    'The person\'s cholesterol level in mg/dl.',
                    'Whether the person\'s fasting blood sugar level is greater than 120 mg/dl (1 if true, 0 if false).',
                    'The results of the person\'s electrocardiogram.',
                    'The person\'s maximum heart rate achieved during exercise.',
                    'The ST depression induced by exercise relative to rest.',
                    'The slope of the ST segment on the EKG.',
                    'The person\'s type of work (e.g., Private, Self-employed, Govt_job).',
                    'The person\'s smoking status (e.g., never smoked, Unknown, smokes, formely smoked).',
                    'Whether the person has heart disease (1 if true, 0 if false).']
}

describe_df = pd.DataFrame(data)
def predict(X_test, scale_type):
    df = pd.read_csv("PreProcessed_Heart_Disease.csv")
    X_train = df.iloc[:,:-1].values
    y_train = df.iloc[:,-1].values
    if scale_type == 'Min Max Scaler':
        scaler = MinMaxScaler()
        scaler.fit_transform(X_train)
        
    elif scale_type == 'Standard Scaler':
        scaler = StandardScaler()
        scaler.fit_transform(X_train)
    
    

    random_forest = RandomForestClassifier(max_depth = 5, n_estimators = 1000)
    random_forest.fit(X_train,y_train)
    return random_forest.predict(X_test)
    
def scale(scale_type, input_df, dataset):
    
    X = dataset.iloc[:,:-1]
    if scale_type == 'Min Max Scaler':
        scaler = MinMaxScaler()
    
    elif scale_type == 'Standard Scaler':
        scaler = StandardScaler()
    
    else:
        return input_df
    
    scaler.fit(X)
    return scaler.transform(input_df)
    
def validate_age(age):
    if age < 0 or age > 120:
        st.error("Please enter a valid age between 0 and 120.")
        return False
    return True

def validate_bp(bp):
    if bp < 0 or bp > 200:
        st.error("Please enter a valid blood pressure between 0 and 300.")
        return False
    return True

def validate_max_hr(max_hr):
    if max_hr < 50 or max_hr > 310:
        st.error("Please enter a valid maximum heart rate between 50 and 230.")
        return False
    return True


def create_df(age, chest_pain_type, bp, cholesterol, fbs, ekg, hr, st_dep, st_slope, Gender, smoking_status, work_type):

    # Create a DataFrame from the data
    input_df = pd.DataFrame.from_dict({'Age': [age],
                    'Gender': [Gender],
                    'Chest pain type': [chest_pain_type],
                    'BP': [bp],
                    'Cholesterol': [cholesterol],
                    'FBS over 120': [fbs],
                    'EKG results': [ekg],
                    'Max HR': [hr],
                    'ST depression': [st_dep],
                    'Slope of ST': [st_slope],
                    'work_type': [work_type],
                    'smoking_status': [smoking_status]})
    
    # Encode the categorical columns
    input_df['Gender'] = input_df['Gender'].replace({'Female': 0, 'Male': 1})
    input_df['work_type'] = input_df['work_type'].replace({'Private': 2, 'self-employed': 3, 'Govt_job': 0, 'children': 4, 'Never_worked': 1})
    input_df['smoking_status'] = input_df['smoking_status'].replace({'never smoked': 2, 'formerly smoked': 1, 'Unknown': 0, 'smokes': 3})

    # Convert the categorical columns to integers
    input_df['Gender'] = input_df['Gender'].astype('int')
    input_df['work_type'] = input_df['work_type'].astype('int')
    input_df['smoking_status'] = input_df['smoking_status'].astype('int')

    # Return the DataFrame
    return input_df

def app():
    st.image("Home_icon.png")
    
    st.title("Columns description.")
    st.table(describe_df)
    st.title("Input")
    
    age = st.number_input("Age", min_value=0, max_value=120)
    if not validate_age(age):
        return
    
    chest_pain_type = st.selectbox("Chest Pain Type", [1,2,3,4])
    
    bp = st.number_input("Blood Pressure", min_value=0, max_value=300)
    if not validate_bp(bp):
        return
    
    cholesterol = st.number_input("Cholesterol")
    
    fbs_over_120 = st.selectbox("FBS Over 120", [0,1])
    
    ekg_results = st.selectbox("EKG Results", [0,1,2])
    
    Gender = st.selectbox("Gender", ['Male','Female'])
    
    smoking_status = st.selectbox("Smoking status", ['never smoked','formerly smoked','Unknown','smokes'])
    
    work_type = st.selectbox("work type", ['Private','Self-employed','Govt_job','children','Never_worked'])
    
    max_hr = st.number_input("Max HR", min_value=50, max_value=230)
    
    if not validate_max_hr(max_hr):
        return
    
    st_depression = st.selectbox("ST Depression", [0,1,2,3,4])
    
    slope_of_st = st.selectbox("Slope of ST", [1,2,3])
    
    st.title("Predictions:")
    scale_type = st.selectbox("scale type", ['None','Min Max Scaler','Standard Scaler'])
    
    if st.button("Submit"):
        if age and bp and max_hr:
            st.success("All inputs are selected.")
            st.text("This model runs with random forest.")
            
            created_df = create_df(age,chest_pain_type,bp,cholesterol,fbs_over_120,ekg_results,max_hr,st_depression,slope_of_st,Gender,smoking_status,work_type)
            scaled_df = scale(scale_type,created_df,PreProcessed_df)
            if predict(scaled_df, scale_type)[0]==0:
                st.markdown("<h1 style='color: green;'>Doesn't have heart disease.</h1>", unsafe_allow_html=True)

            else:
                st.markdown("<h1 style='color: red;'>have heart disease.</h1>", unsafe_allow_html=True)
                
        else:
            st.error("Please select all inputs.")
if __name__ == '__main__':
    app()