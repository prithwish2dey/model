from flask import Flask, request, jsonify, render_template, Response
import numpy as np
import joblib
import pandas as pd
import logging

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model for disease prediction
disease_model_path = './models/disease_prediction/random_forest.joblib'
disease_loaded_model = joblib.load(disease_model_path)

# Load the pre-trained model for time prediction
time_model_path = './models/time_prediction/random_forest_model.pkl'
time_loaded_model = joblib.load(time_model_path)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Route for rendering the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Route for predicting disease and time
@app.route('/predict_disease_and_time', methods=['POST'])
def predict_disease_and_time():
    try:
        # Parse input JSON
        input_data = request.get_json()
        logging.debug(f"Input data received: {input_data}")

        symptoms = {'itching': 0, 'skin_rash': 0, 'nodal_skin_eruptions': 0, 'continuous_sneezing': 0,
                'shivering': 0, 'chills': 0, 'joint_pain': 0, 'stomach_pain': 0, 'acidity': 0, 'ulcers_on_tongue': 0,
                'muscle_wasting': 0, 'vomiting': 0, 'burning_micturition': 0, 'spotting_ urination': 0, 'fatigue': 0,
                'weight_gain': 0, 'anxiety': 0, 'cold_hands_and_feets': 0, 'mood_swings': 0, 'weight_loss': 0,
                'restlessness': 0, 'lethargy': 0, 'patches_in_throat': 0, 'irregular_sugar_level': 0, 'cough': 0,
                'high_fever': 0, 'sunken_eyes': 0, 'breathlessness': 0, 'sweating': 0, 'dehydration': 0,
                'indigestion': 0, 'headache': 0, 'yellowish_skin': 0, 'dark_urine': 0, 'nausea': 0, 'loss_of_appetite': 0,
                'pain_behind_the_eyes': 0, 'back_pain': 0, 'constipation': 0, 'abdominal_pain': 0, 'diarrhoea': 0, 'mild_fever': 0,
                'yellow_urine': 0, 'yellowing_of_eyes': 0, 'acute_liver_failure': 0, 'fluid_overload': 0, 'swelling_of_stomach': 0,
                'swelled_lymph_nodes': 0, 'malaise': 0, 'blurred_and_distorted_vision': 0, 'phlegm': 0, 'throat_irritation': 0,
                'redness_of_eyes': 0, 'sinus_pressure': 0, 'runny_nose': 0, 'congestion': 0, 'chest_pain': 0, 'weakness_in_limbs': 0,
                'fast_heart_rate': 0, 'pain_during_bowel_movements': 0, 'pain_in_anal_region': 0, 'bloody_stool': 0,
                'irritation_in_anus': 0, 'neck_pain': 0, 'dizziness': 0, 'cramps': 0, 'bruising': 0, 'obesity': 0, 'swollen_legs': 0,
                'swollen_blood_vessels': 0, 'puffy_face_and_eyes': 0, 'enlarged_thyroid': 0, 'brittle_nails': 0, 'swollen_extremeties': 0,
                'excessive_hunger': 0, 'extra_marital_contacts': 0, 'drying_and_tingling_lips': 0, 'slurred_speech': 0,
                'knee_pain': 0, 'hip_joint_pain': 0, 'muscle_weakness': 0, 'stiff_neck': 0, 'swelling_joints': 0, 'movement_stiffness': 0,
                'spinning_movements': 0, 'loss_of_balance': 0, 'unsteadiness': 0, 'weakness_of_one_body_side': 0, 'loss_of_smell': 0,
                'bladder_discomfort': 0, 'foul_smell_of urine': 0, 'continuous_feel_of_urine': 0, 'passage_of_gases': 0, 'internal_itching': 0,
                'toxic_look_(typhos)': 0, 'depression': 0, 'irritability': 0, 'muscle_pain': 0, 'altered_sensorium': 0,
                'red_spots_over_body': 0, 'belly_pain': 0, 'abnormal_menstruation': 0, 'dischromic _patches': 0, 'watering_from_eyes': 0,
                'increased_appetite': 0, 'polyuria': 0, 'family_history': 0, 'mucoid_sputum': 0, 'rusty_sputum': 0, 'lack_of_concentration': 0,
                'visual_disturbances': 0, 'receiving_blood_transfusion': 0, 'receiving_unsterile_injections': 0, 'coma': 0,
                'stomach_bleeding': 0, 'distention_of_abdomen': 0, 'history_of_alcohol_consumption': 0, 'fluid_overload.1': 0,
                'blood_in_sputum': 0, 'prominent_veins_on_calf': 0, 'palpitations': 0, 'painful_walking': 0, 'pus_filled_pimples': 0,
                'blackheads': 0, 'scurring': 0, 'skin_peeling': 0, 'silver_like_dusting': 0, 'small_dents_in_nails': 0, 'inflammatory_nails': 0,
                'blister': 0, 'red_sore_around_nose': 0, 'yellow_crust_ooze': 0}

        # Update symptom values based on input data
        for symptom in input_data.get('symptoms', []):
            if symptom in symptoms:
                symptoms[symptom] = 1
        
        # Put all data in a test dataset
        df_test = pd.DataFrame(columns=list(symptoms.keys()))
        df_test.loc[0] = np.array(list(symptoms.values()))
        
        # Make disease prediction
        disease_prediction = disease_loaded_model.predict(df_test)
        logging.debug(f"Disease: {disease_prediction[0]}")


        disease_codes = {
        "Fungal infection": 1,
        "Allergy": 2,
        "GERD": 3,
        "Chronic cholestasis": 4,
        "Drug Reaction": 5,
        "Peptic ulcer disease": 6,
        "AIDS": 7,
        "Diabetes": 8,
        "Gastroenteritis": 9,
        "Bronchial Asthma": 10,
        "Hypertension": 11,
        "Migraine": 12,
        "Cervical spondylosis": 13,
        "Paralysis (brain hemorrhage)": 14,
        "Jaundice": 15,
        "Malaria": 16,
        "Chicken pox": 17,
        "Dengue": 18,
        "Typhoid": 19,
        "Hepatitis A": 20,
        "Hepatitis B": 21,
        "Hepatitis C": 22,
        "Hepatitis D": 23,
        "Hepatitis E": 24,
        "Alcoholic hepatitis": 25,
        "Tuberculosis": 26,
        "Common Cold": 27,
        "Pneumonia": 28,
        "Dimorphic hemorrhoids(piles)": 29,
        "Heart attack": 30,
        "Varicose veins": 31,
        "Hypothyroidism": 32,
        "Hyperthyroidism": 33,
        "Hypoglycemia": 34,
        "Osteoarthritis": 35,
        "Arthritis": 36,
        "(vertigo) Paroxysmal Positional Vertigo": 37,
        "Acne": 38,
        "Urinary tract infection": 39,
        "Psoriasis": 40,
        "Impetigo": 41
        }


        # Get the disease code from the Excel file
        disease_code = disease_codes[disease_prediction[0]]
        logging.debug(f"Disease Code: {disease_code}")

        # Extract other input values for time prediction
        visit_number = input_data["visitNumber"]
        patient_age = input_data["patientAge"]
        patient_sex = input_data["patientSex"]
        
        # Prepare features for time prediction
        feature_values = [
            visit_number,
            patient_age,
            patient_sex,
            disease_code
        ]
        
        # Convert to NumPy array for prediction
        new_data_point = np.array([feature_values])

        # Make time prediction
        predicted_time = time_loaded_model.predict(new_data_point)
        logging.debug(f"Predicted Time: {predicted_time[0]:.2f}")

        # Return the results as JSON
        return jsonify({
            # "Predicted Disease": disease_prediction,
            "Predicted Time Taken (seconds)": round(predicted_time[0], 2)
        })
    
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
