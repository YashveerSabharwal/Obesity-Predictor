import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Obesity Detector")
st.markdown("---") 
st.subheader("Yashveer Sabharwal, 2024B3PS1000G")

tab1,tab2,tab3 = st.tabs(['Predict IT','Base Models','Ensemble Learning Methods'])

with tab1:
     # Streamlit App Title
    st.title("Obesity Prediction Form")
    st.write("my submission is not working will work on that after and with physical guidance")

    # 1. Input fields
    gender = st.selectbox("Gender", ['Male', 'Female'])
    age = st.number_input("Age", min_value=0, max_value=120)
    height = st.number_input("Height (in meters)", min_value=0.5, max_value=2.5)
    weight = st.number_input("Weight (in kg)", min_value=10.0, max_value=250.0)

    family_history = st.selectbox("Family History with Overweight", ['yes', 'no'])
    favc = st.selectbox("Frequent High Calorie Food Consumption (FAVC)", ['yes', 'no'])
    fcvc = st.slider("Frequency of Vegetable Consumption (1 = low, 3 = high)", 1.0, 3.0, 2.0)
    ncp = st.slider("Number of Main Meals per Day", 1.0, 5.0, 3.0)
    caec = st.selectbox("Consumption of Between-Meal Snacks (CAEC)", ['no', 'Sometimes', 'Frequently', 'Always'])
    smoke = st.selectbox("Do You Smoke?", ['no', 'yes'])
    ch2o = st.slider("Daily Water Intake (CH2O)", 1.0, 3.0, 2.0)
    scc = st.selectbox("Monitor Caloric Consumption (SCC)", ['no', 'yes'])
    faf = st.slider("Physical Activity Frequency (FAF)", 0.0, 3.0, 1.0)
    tue = st.slider("Time Using Technology Devices (TUE)", 0.0, 2.0, 1.0)
    calc = st.selectbox("Alcohol Consumption (CALC)", ['no', 'Sometimes', 'Frequently'])
    mtrans = st.selectbox("Mode of Transportation", ['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'])

    # 2. Encoding categorical values
    gender = 0 if gender == 'Male' else 1
    family_history = 1 if family_history == 'yes' else 0
    favc = 1 if favc == 'yes' else 0
    caec = ['no', 'Sometimes', 'Frequently', 'Always'].index(caec)
    smoke = 1 if smoke == 'yes' else 0
    scc = 1 if scc == 'yes' else 0
    calc = ['no', 'Sometimes', 'Frequently'].index(calc)
    mtrans = ['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'].index(mtrans)

    # 3. Build DataFrame for model
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'family_history_with_overweight': [family_history],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'CAEC': [caec],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'SCC': [scc],
        'FAF': [faf],
        'TUE': [tue],
        'CALC': [calc],
        'MTRANS': [mtrans]
    })

    st.subheader("Model Input Preview")
    st.dataframe(input_data)

    # You can now pass `input_data` to your model like:
    # prediction = model.predict(input_data)
    import pickle


    # Define model and algorithm names
    algonames = [ 'Random Forest','Soft Voting','Bagging','Bagged Decision Tree']
    modelnames = [ 'Randomforest (1).pkl','SoftVoting.pkl','Bagging.pkl','BaggedDT.pkl']

    # Class labels
    obesity_classes = [
        'Overweight_Level_II',     # 0
        'Normal_Weight',           # 1
        'Insufficient_Weight',     # 2
        'Obesity_Type_III',        # 3
        'Obesity_Type_II',         # 4
        'Overweight_Level_I',      # 5
        'Obesity_Type_I'           # 6
    ]

    # Function to predict obesity level
    def predict_obesity(data):
        predictions = []
        for modelname in modelnames:
            with open(modelname, 'rb') as file:
                model = pickle.load(file)
                result = model.predict(data)  # result is array like [6] or [1]
                predictions.append(result[0])  # get the class index
        return predictions

    # Submit button
    if st.button("Submit"):
        st.subheader("Predicted Obesity Category")
        st.markdown("------")

        results = predict_obesity(input_data)

        for i, prediction in enumerate(results):
            st.subheader(algonames[i])
            st.write(f"Prediction: **{obesity_classes[prediction]}**")
            st.markdown("------")


with tab2:
   

    data = {
        'Decision Trees': 86.77,
        'Logistic Regression': 84.15,
        'Support Vector Machine': 79.48
    }

    Models = list(data.keys())
    Accuracies = list(data.values())

    df = pd.DataFrame(list(zip(Models, Accuracies)), columns=['Models', 'Accuracies'])
    fig = px.bar(df, x='Models', y='Accuracies')
    st.plotly_chart(fig)

with tab3:
    
    data = {
        'Random Forest': 76.78,
        'Bagging': 86.91,
        'Boosting': 76.65,
        'Soft Voting': 83.88,
        'Hard Voting': 48.91,
        'Weighted Voting': 83.62,
    }

    Models = list(data.keys())
    Accuracies = list(data.values())

    df = pd.DataFrame(list(zip(Models, Accuracies)), columns=['Models', 'Accuracies'])
    fig = px.bar(df, x='Models', y='Accuracies')
    st.plotly_chart(fig)
     
