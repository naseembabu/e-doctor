# import necessary libraries
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Loading models
dia_model = pickle.load(open("./savedModels/Diabetes.sav", 'rb'))
heart_model = pickle.load(open("./savedModels/Heart.sav", 'rb'))
par_model = pickle.load(open("./savedModels/Parkinsons.sav", 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('E- Doctor System',
                           ['Diabetes Screening',
                            'Heart Health Screening',
                            'Parkinsons Screening'],
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Screening':
    # page title
    st.title('Provide Information for Diabetes Prediction')
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        SkinThickness = st.text_input('Skin Thickness value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Glucose = st.text_input('Glucose Level')
        Insulin = st.text_input('Insulin Level')
        BMI = st.text_input('BMI value')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
        Age = st.text_input('Age of the Person')

    # code for Prediction
    diab_diagnosis = ''
    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        diab_prediction = dia_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Health Screening':
    # page title
    st.title('Provide Information for Heart Analysis')
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
        trestbps = st.text_input('Resting Blood Pressure')
        restecg = st.text_input('Resting Electrocardiographic results')
        exang = st.text_input('Exercise Induced Angina')
        slope = st.text_input('Slope of the peak exercise ST segment')
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
    with col2:
        sex = st.text_input('Sex')
        chol = st.text_input('Serum Cholestoral in mg/dl')
        thalach = st.text_input('Maximum Heart Rate achieved')
        oldpeak = st.text_input('ST depression induced by exercise')
        ca = st.text_input('Major vessels colored by flourosopy')
    with col3:
        cp = st.text_input('Chest Pain types')
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        APQ3 = st.text_input('Shimmer:APQ3')
        APQ5 = st.text_input('Shimmer:APQ5')
        APQ = st.text_input('MDVP:APQ')

    # code for Prediction
    heart_diagnosis = ''
    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Screening":
    # page title
    st.title("Provide Information for Parkinsons Analysis")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        RAP = st.text_input('MDVP:RAP')
        Shimmer = st.text_input('MDVP:Shimmer')
        APQ = st.text_input('MDVP:APQ')
        D2 = st.text_input('D2')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        PPQ = st.text_input('MDVP:PPQ')
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        DDA = st.text_input('Shimmer:DDA')
        PPE = st.text_input('PPE')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        DDP = st.text_input('Jitter:DDP')
        APQ3 = st.text_input('Shimmer:APQ3')
        NHR = st.text_input('NHR')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        Shimmer = st.text_input('MDVP:Shimmer')
        APQ5 = st.text_input('Shimmer:APQ5')
        HNR = st.text_input('HNR')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        RPDE = st.text_input('RPDE')
        APQ = st.text_input('MDVP:APQ')
        spread1 = st.text_input('spread1')
        spread2 = st.text_input('spread2')

    # code for Prediction
    parkinsons_diagnosis = ''
    # creating a button for Prediction
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = par_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

