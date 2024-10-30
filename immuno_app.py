import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the scaler and model
scaler = joblib.load('preprocessor.joblib')
model = joblib.load('QML_model.joblib')

st.write("""
# IMMUNOTHERAPY RESPONSE PREDICTION
""")

# Add file upload option
st.write("""
## Choose Input Method
""")
input_method = st.radio(
    "Select how you want to input patient data:",
    ('Single Patient Form', 'Upload Dataset')
)

def user_input_features():
    # Define cancer type options
    cancer_types = ['NSCLC', 'Melanoma', 'Renal', 'Bladder', 'Head & Neck', 
                   'Sarcoma', 'Endometrial', 'Gastric', 'Hepatobiliary', 'SCLC',
                   'Colorectal', 'Esophageal', 'Pancreatic', 'Mesothelioma',
                   'Ovarian', 'Breast']
    
    cancer_type_2_options = ['Others', 'NSCLC', 'Melanoma']

    drug_class_options = ['PD-1/PD-L1', 'CTLA-4', 'Combination']
    
    # Sidebar inputs
    cancer_type_grouped_2 = st.sidebar.selectbox(
        'Cancer Type Grouped 2',
        options=cancer_types,
        help="Specific cancer type classification"
    )
    
    cancer_type = st.sidebar.selectbox(
        'Cancer Type',
        options=cancer_type_2_options,
        help="General cancer type classification"
    )
    
    chemo_before_io = st.sidebar.number_input(
        'Chemo_before_IO (1:Yes; 0:No)', min_value=0, max_value=1,
        help="Chemotherapy before Immunotherapy (1 = Yes, 0 = No)"
    )
    
    age = st.sidebar.number_input(
        'Age', min_value=0,
        help="Age of the patient in years"
    )
    
    sex = st.sidebar.number_input(
        'Sex (1:Male; 0:Female)', min_value=0, max_value=1,
        help="Gender of the patient (1 = Male, 0 = Female)"
    )
    
    bmi = st.sidebar.number_input(
        'BMI', min_value=0.0,
        help="Body Mass Index"
    )
    
    stage = st.sidebar.number_input(
        'Stage (1:IV; 0:I-III)', min_value=0, max_value=1,
        help="Cancer Stage (1 = Stage IV, 0 = Stage I-III)"
    )
    
    stage_io_start_options = ['I', 'II', 'III', 'IV']
    stage_io_start = st.sidebar.selectbox(
        'Stage at IO start',
        options=stage_io_start_options,
        help="Stage of cancer when immunotherapy was initiated"
    )
    
    
    nlr = st.sidebar.number_input(
        'NLR', min_value=0.0,
        help="Neutrophil-to-Lymphocyte Ratio"
    )
    
    platelets = st.sidebar.number_input(
        'Platelets', min_value=0.0,
        help="Platelet count"
    )
    
    hgb = st.sidebar.number_input(
        'HGB', min_value=0.0,
        help="Hemoglobin level"
    )
    
    albumin = st.sidebar.number_input(
        'Albumin', min_value=0.0,
        help="Albumin level"
    )
    
    drug = st.sidebar.number_input(
        'Drug (1:Combo; 0:PD1/PDL1orCTLA4)', min_value=0, max_value=1,
        help="Drug Type (1 = Combination Therapy, 0 = PD-1/PD-L1 or CTLA-4 Therapy)"
    )
    
    drug_class = st.sidebar.selectbox(
        'Drug_class',
        options=drug_class_options,
        help="Classification of the drug used"
    )
    
    tmb = st.sidebar.number_input(
        'TMB', min_value=0.0,
        help="Tumor Mutational Burden"
    )
    
    fcna = st.sidebar.number_input(
        'FCNA', min_value=0.0,
        help="Fraction of Copy Number Alterations"
    )
    
    hed = st.sidebar.number_input(
        'HED', min_value=0.0,
        help="Homologous Recombination Deficiency"
    )
    
    hla_loh = st.sidebar.number_input(
        'HLA_LOH', min_value=0.0,
        help="HLA Loss of Heterozygosity"
    )
    
    msi = st.sidebar.number_input(
        'MSI (1:Unstable; 0:Stable_Indeterminate)', min_value=0, max_value=1,
        help="Microsatellite Instability (1 = Unstable, 0 = Stable or Indeterminate)"
    )
    
    msi_score = st.sidebar.number_input(
        'MSI_SCORE', min_value=0.0,
        help="Microsatellite Instability Score"
    )
    

    os_event = st.sidebar.number_input(
        'OS_Event', min_value=0, max_value=1,
        help="Overall Survival Event"
    )
    
    os_months = st.sidebar.number_input(
        'OS_Months', min_value=0.0,
        help="Overall Survival in Months"
    )
    
    pfs_event = st.sidebar.number_input(
        'PFS_Event', min_value=0, max_value=1,
        help="Progression-Free Survival Event"
    )
    
    pfs_months = st.sidebar.number_input(
        'PFS_Months', min_value=0.0,
        help="Progression-Free Survival in Months"
    )
    
    rf16_prob = st.sidebar.number_input(
        'RF16_prob', min_value=0.0,
        help="Random Forest Model Probability"
    )
    
    data = {
        'Cancer_type_grouped_2': cancer_type_grouped_2,
        'Cancer_Type': cancer_type,
        'Chemo_before_IO (1:Yes; 0:No)': chemo_before_io,
        'Age': age,
        'Sex (1:Male; 0:Female)': sex,
        'BMI': bmi,
        'Stage (1:IV; 0:I-III)': stage,
        'Stage at IO start': stage_io_start,
        'NLR': nlr,
        'Platelets': platelets,
        'HGB': hgb,
        'Albumin': albumin,
        'Drug (1:Combo; 0:PD1/PDL1orCTLA4)': drug,
        'Drug_class': drug_class,
        'TMB': tmb,
        'FCNA': fcna,
        'HED': hed,
        'HLA_LOH': hla_loh,
        'MSI (1:Unstable; 0:Stable_Indeterminate)': msi,
        'MSI_SCORE': msi_score,
        'OS_Event': os_event,
        'OS_Months': os_months,
        'PFS_Event': pfs_event,
        'PFS_Months': pfs_months,
        'RF16_prob': rf16_prob
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

def process_predictions(df):
    # Apply scaling
    df_scaled = scaler.transform(df)
    
    # Make predictions
    prediction_probas = model.predict_proba(df_scaled)
    
    # Determine prediction based on probability threshold (0.5)
    # If probability of being a responder (class 1) is > 0.5, predict responder
    predictions = (prediction_probas[:, 1] > 0.5).astype(bool)
    
    # Create results DataFrame
    results_df = df.copy()
    results_df['Prediction'] = ['Responder' if pred else 'Non-responder' for pred in predictions]
    results_df['Probability_Non_Responder'] = prediction_probas[:, 0]
    results_df['Probability_Responder'] = prediction_probas[:, 1]
    
    return results_df

if input_method == 'Single Patient Form':
    st.sidebar.header('User Input Parameters')
    df = user_input_features()
    
    st.subheader('User Input Parameters')
    st.write(df)
    
    results_df = process_predictions(df)
    
    st.subheader('Prediction')
    st.write(results_df['Prediction'][0])
    
    st.subheader('Prediction Probabilities')
    col1, col2 = st.columns(2)
    with col1:
        st.write('Non-responder probability:', results_df['Probability_Non_Responder'][0])
    with col2:
        st.write('Responder probability:', results_df['Probability_Responder'][0])

else:
    st.write("""
    ## Upload Dataset
    Please upload a CSV file containing patient data. The file should include all required columns with the same names as shown in the single patient form.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the dataset
            input_df = pd.read_csv(uploaded_file)
            
            # Check if all required columns are present
            required_columns = list(user_input_features().columns)
            missing_columns = [col for col in required_columns if col not in input_df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                st.subheader('Uploaded Dataset Preview')
                st.write(input_df.head())
                
                # Process predictions for all patients
                results_df = process_predictions(input_df)
                
                st.subheader('Prediction Results')
                st.write(results_df)
                
                # Add download button for results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="immunotherapy_predictions.csv",
                    mime="text/csv"
                )
                
                # Show summary statistics
                st.subheader('Summary Statistics')
                total_patients = len(results_df)
                responders = (results_df['Prediction'] == 'Responder').sum()
                non_responders = total_patients - responders
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Patients", total_patients)
                with col2:
                    st.metric("Predicted Responders", responders)
                with col3:
                    st.metric("Predicted Non-responders", non_responders)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")