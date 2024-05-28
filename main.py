import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import streamlit.components.v1 as components

# Load HTML files
def load_html(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        return file.read()

# Load the models and scalers
try:
    diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
    parkinson_model = pickle.load(open('parkinson_model.sav', 'rb'))
    heart_model = pickle.load(open('heart_model.sav', 'rb'))
    scaler_d = pickle.load(open('scaler_d.sav', 'rb'))
    scaler_p = pickle.load(open('scaler.sav', 'rb'))
    scaler_h = pickle.load(open('scaler_heart.sav', 'rb'))

except Exception as e:
    st.error(f"Error loading models or scalers: {e}")
    st.stop()

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f4f4f9;
        color: #333;
        font-family: Arial, sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #a6dcef;
    }
    h1 {
        color: #003366;
    }
    h3 {
        color: #003366;
    }
    .stButton>button {
        background-color: #003366;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #00509e;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    selected = option_menu(
        "Wellness Predict",
        ["Home", "Healthy Lifestyle Tips", "Diabetes Prediction", "Heart Prediction", "Parkinson's Prediction"],
        icons=['house', 'heart', 'activity', 'heart', 'person'],
        default_index=0
    )

# Home page
if selected == "Home":
    with open("index.html", "r") as f:
        home_content = f.read()
    st.markdown(home_content, unsafe_allow_html=True)

# Healthy Lifestyle Tips page
elif selected == "Healthy Lifestyle Tips":
    with open("life.html", "r") as f:
        home_content = f.read()
    st.markdown(home_content, unsafe_allow_html=True)

# Diabetes Prediction page
elif selected == "Diabetes Prediction":
    st.markdown("<h1>Diabetes Prediction</h1>", unsafe_allow_html=True)
    st.markdown("Enter the required information below:")

    with st.form("diabetes_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
            Glucose = st.number_input('Glucose Level', min_value=0.0, step=0.00001)
            BloodPressure = st.number_input('Blood Pressure', min_value=0.0, step=0.00001)
            SkinThickness = st.number_input('Skin Thickness', min_value=0.0, step=0.00001)
        with col2:
            Insulin = st.number_input('Insulin Level', min_value=0.0, step=0.00001)
            BMI = st.number_input('BMI', min_value=0.0, step=0.00001)
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, step=0.00001)
            Age = st.number_input('Age of the Person', min_value=0, step=1)

        if st.form_submit_button("Predict"):
            try:
                user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
                input_data = np.array(user_input).reshape(1, -1)
                std_data = scaler_d.transform(input_data)
                diab_prediction = diabetes_model.predict(std_data)
                if diab_prediction[0] == 1:
                    diab_diagnosis = 'The person is diabetic'
                    st.error(diab_diagnosis)
                else:
                    diab_diagnosis = 'The person is not diabetic'
                    st.success(diab_diagnosis)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Heart Prediction page
elif selected == "Heart Prediction":
    st.title("Heart Health Prediction")
    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # Code for Prediction
    heart_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Heart Test Result'):
        try:
            # Convert input data to float and ensure all inputs are numeric
            user_input = [float(age), float(sex), float(cp), 
                          float(trestbps), float(chol), float(fbs), 
                          float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]
            
            # Log the user input
            # st.write("User input values: ", user_input)
            
            # Ensure the input is in a 2D array (for a single sample)
            input_data = np.array(user_input).reshape(1, -1)
            
            # Scale the input data using the loaded scaler
            std_data = scaler_h.transform(input_data)
            
            # Make prediction
            heart_prediction = heart_model.predict(std_data)
            
            # Log the prediction result
            # st.write("Model prediction: ", heart_prediction)
            
            # If the model supports probability output, print it
            if hasattr(diabetes_model, "predict_proba"):
                diab_prediction_proba = diabetes_model.predict_proba(std_data)
                st.write("Prediction probabilities: ", diab_prediction_proba)
            
            # Check prediction and display result
            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person has heart disease'
                st.error(heart_diagnosis)
            else:
                heart_diagnosis = 'The person has normal condition'
                st.success(heart_diagnosis)
            
            
        except ValueError as e:
            st.error(f"Please enter valid numerical values. Error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Parkinson's Prediction page
elif selected == "Parkinson's Prediction":
    st.title("Parkinson's Prediction")
    
    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        MDVP_Fo_Hz = st.text_input('MDVP Fo (Hz)')

    with col2:
        MDVP_Fhi_Hz = st.text_input('MDVP Fhi (Hz)')

    with col3:
        MDVP_Flo_Hz = st.text_input('MDVP Flo (Hz)')

    with col1:
        MDVP_Jitter_percentage = st.text_input('MDVP Jitter (%)')

    with col2:
        MDVP_Jitter_Abs = st.text_input('MDVP Jitter (Abs)')

    with col3:
        MDVP_RAP = st.text_input('MDVP RAP')

    with col1:
        MDVP_PPQ = st.text_input('MDVP PPQ')

    with col2:
        Jitter_DDP = st.text_input('Jitter DDP')

    with col3:
        MDVP_Shimmer = st.text_input('MDVP Shimmer')

    with col1:
        MDVP_Shimmer_dB = st.text_input('MDVP Shimmer (dB)')

    with col2:
        Shimmer_APQ3 = st.text_input('Shimmer APQ3')

    with col3:
        Shimmer_APQ5 = st.text_input('Shimmer APQ5')

    with col1:
        MDVP_APQ = st.text_input('MDVP APQ')

    with col2:
        Shimmer_DDA = st.text_input('Shimmer DDA')

    with col3:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col1:
        spread1 = st.text_input('spread1')

    with col2:
        spread2 = st.text_input('spread2')
    
    with col3:
        D2 = st.text_input('D2')

    with col1:
        PPE = st.text_input('PPE')

    # Code for Prediction
    parkinson_diagnosis = ''

    # Creating a button for Prediction
    if st.button("Predict Parkinson's"):
        try:
            # Convert input data to float and ensure all inputs are numeric
            user_input = [float(MDVP_Fo_Hz), float(MDVP_Fhi_Hz), float(MDVP_Flo_Hz), 
                          float(MDVP_Jitter_percentage), float(MDVP_Jitter_Abs), float(MDVP_RAP), 
                          float(MDVP_PPQ), float(Jitter_DDP), float(MDVP_Shimmer), 
                          float(MDVP_Shimmer_dB), float(Shimmer_APQ3), float(Shimmer_APQ5), 
                          float(MDVP_APQ), float(Shimmer_DDA), float(NHR), float(HNR), 
                          float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]
            
            # Log the user input
            # st.write("User input values: ", user_input)
            
            # Ensure the input is in a 2D array (for a single sample)
            input_data = np.array(user_input).reshape(1, -1)
            std_data = scaler_p.transform(input_data)
            
            # Make prediction
            parkinson_prediction = parkinson_model.predict(std_data)
            
            # Log the prediction result
            # st.write("Model prediction: ", parkinson_prediction)
            
            # If the model supports probability output, print it
            if hasattr(parkinson_model, "predict_proba"):
                parkinson_prediction_proba = parkinson_model.predict_proba(input_data)
                st.write("Prediction probabilities: ", parkinson_prediction_proba)
            
            # Check prediction and display result
            if parkinson_prediction[0] == 1:
                parkinson_diagnosis = 'The person has Parkinsons'
                st.error(parkinson_diagnosis)
            else:
                parkinson_diagnosis = 'The person does not have Parkinsons'
                st.success(parkinson_diagnosis)
            
            
        except ValueError as e:
            st.error(f"Please enter valid numerical values. Error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
