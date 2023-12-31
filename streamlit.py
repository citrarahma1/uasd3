import pickle
import streamlit as st
from dtreeviz.trees import dtreeviz

model = pickle.load(open('brain_stroke.sav', 'rb'))

st.title('Kalkulasi Resiko Terkena Stroke Dengan Algoritma Decision Tree')

gender = st.selectbox('Jenis Kelamin anda ?', ['Laki laki', 'Perempuan'])

if gender == 'Laki laki':
    gender = 1
else:
    gender = 0
age = st.text_input('Usia Anda')
hypertension = st.selectbox('Apakah anda memiliki darah tinggi', ['Ya', 'Tidak'])

if hypertension == 'Ya':
    hypertension = 1
else:
    hypertension = 0
heart_disease = st.selectbox('Apakah anda memiliki penyakit jantung', ['Ya', 'Tidak'])

if heart_disease == 'Ya':
    heart_disease = 1
else:
    heart_disease = 0
ever_married = st.selectbox('Apakah anda pernah menikah', ['Ya', 'Tidak'])

if ever_married == 'Ya':
    ever_married = 1
else:
    ever_married = 0
avg_glucose_level = st.text_input('Rata rata kadar gula darah')
bmi = st.text_input('Nilai Body Mass Index')
work_type_Govt_job = st.selectbox('Apakah anda bekerja di instansi pemerintahan ?', ['Ya', 'Tidak'])

if work_type_Govt_job == 'Ya':
    work_type_Govt_job = 1
else:
    work_type_Govt_job = 0
work_type_Private = st.selectbox('Apakah anda bekerja di sebuah perushaan ?', ['Ya', 'Tidak'])

if work_type_Private == 'Ya':
    work_type_Private = 1
else:
    work_type_Private = 0

work_type_Self_employed = st.selectbox('Apakah anda bekerja secara Freelance ?', ['Ya', 'Tidak'])

if work_type_Self_employed == 'Ya':
    work_type_Self_employed = 1
else:
    work_type_Self_employed = 0

work_type_children = st.selectbox('Apakah anda bekerja disekitar anak anak ?', ['Ya', 'Tidak'])

if work_type_children == 'Ya':
    work_type_children = 1
else:
    work_type_children = 0

Residence_type_Rural = st.selectbox('Apakah anda tinggal di pedesaan ?', ['Ya', 'Tidak'])

if Residence_type_Rural == 'Ya':
    Residence_type_Rural = 1
else:
    Residence_type_Rural = 0

Residence_type_Urban = st.selectbox('Apakah anda tinggal di perkotaan ?', ['Ya', 'Tidak'])

if Residence_type_Urban == 'Ya':
    Residence_type_Urban = 1
else:
    Residence_type_Urban = 0

smoking_status_Unknown = st.selectbox('Apakah anda tidak sadar kalo anda seoramg perokok pasif ?', ['Ya', 'Tidak'])

if smoking_status_Unknown == 'Ya':
    smoking_status_Unknown = 1
else:
    smoking_status_Unknown = 0

smoking_status_formerly_smoked = st.selectbox('Apakah anda mantan perokok ?', ['Ya', 'Tidak'])

if smoking_status_formerly_smoked == 'Ya':
    smoking_status_formerly_smoked = 1
else:
    smoking_status_formerly_smoked = 0


smoking_status_never_smoked = st.selectbox('Apakah anda tidak pernah merokok ?', ['Ya', 'Tidak'])

if smoking_status_never_smoked == 'Ya':
    smoking_status_never_smoked = 1
else:
    smoking_status_never_smoked = 0


smoking_status_smokes = st.selectbox('Apakah anda masih merokok ?', ['Ya', 'Tidak'])

if smoking_status_smokes == 'Ya':
    smoking_status_smokes = 1
else:
    smoking_status_smokes = 0


resiko = ''

if st.button('Tingkat Resiko'):
    tingkat_resiko = model.predict([[gender, age, hypertension, heart_disease, ever_married, avg_glucose_level, bmi, work_type_Govt_job, work_type_Private, work_type_Self_employed, work_type_children, Residence_type_Rural, Residence_type_Urban, smoking_status_Unknown, smoking_status_formerly_smoked, smoking_status_never_smoked, smoking_status_smokes]])
    
    if(tingkat_resiko[0] == 0):
        resiko = 'Anda tidak beresiko terkena Stroke'
    else :
        resiko ='Anda beresiko tekena Stroke'

    st.success(resiko)

if st.button('Visualize Decision Tree'):
    # Visualize the Decision Tree using dtreeviz
    viz = dtreeviz(model, X_train, y_train, target_name='target', feature_names=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'bmi', 'stroke', 'work_type_Govt_job', 'work_type_Private', 'work_type_Self_employed', 'work_type_children', 'Residence_type_Rural', 'Residence_type_Urban', 'smoking_status_Unknown', 'smoking_status_formerly_smoked', 'smoking_status_never_smoked', 'smoking_status_smokes'])
    st.pyplot(viz.to_image())
