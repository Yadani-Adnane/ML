import streamlit as st

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df=pd.read_csv("Breast Cancer Wisconsin (Diagnostic) Data.csv")
df.drop(columns=['Unnamed: 32'],inplace=True)
from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])
df = df.drop('id', axis=1)
X = df.drop(columns='diagnosis',axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.fit_transform(X_test)
svm=SVC()
svm.fit(X_train,y_train)

st.write('''
# Classification du Diagnostic du Cancer du Sein à l'Aide de Modèles de Machine Learning
Ce projet vise à développer un système de classification des patients en fonction de leur diagnostic
médical à partir de données médicales anonymisées. Permettant de prédire si une tumeur mammaire est
bénigne ou maligne en utilisant des techniques de machine learning. Les données utilisées pour former
et évaluer les modèles comprendront des mesures cliniques des cellules présentes dans les biopsies de
tumeurs.
''')


st.sidebar.header("Les parametres d'entrée :")
input_data =st.sidebar.text_area('Enter 30 valeurs séparées par des virgules " , ":', height=100)
values = input_data.split(',')
values = [value.strip() for value in values]
if len(values)==30:
    data1 = {
        'radius_mean': values[0],
        'texture_mean': values[1],
        'perimeter_mean': values[2],
        'area_mean': values[3],
        'smoothness_mean': values[4],
        'compactness_mean': values[5],
        'concavity_mean': values[6],
        'concave points_mean': values[7],
        'symmetry_mean': values[8],
        'fractal_dimension_mean': values[9],
        'radius_se': values[10],
        'texture_se': values[11],
        'perimeter_se': values[12],
        'area_se': values[13],
        'smoothness_se': values[14],
        'compactness_se': values[15],
        'concavity_se': values[16],
        'concave points_se': values[17],
        'symmetry_se': values[18],
        'fractal_dimension_se': values[19],
        'radius_worst': values[20],
        'texture_worst': values[21],
        'perimeter_worst': values[22],
        'area_worst': values[23],
        'smoothness_worst': values[24],
        'compactness_worst': values[25],
        'concavity_worst': values[26],
        'concave points_worst': values[27],
        'symmetry_worst': values[28],
        'fractal_dimension_worst': values[29]
    }
if st.sidebar.button('Envoyer'):
    if input_data != '':
        df2 = pd.DataFrame(data1, index=[0])
        svm_pred= svm.predict(df2)
        st.subheader('Données de patient:')
        st.write(df2)
        if svm_pred == 1:
            st.subheader("Le patient est Maligne.")
        else: st.subheader("Le patient est Bénigne,")
    else: st.sidebar.error("Enter 30 valeurs séparées par des virgules!")

st.sidebar.subheader("oubien replir les champs suivants :")
def user_input():
    radius_mean = st.sidebar.number_input("radius_mean")
    texture_mean= st.sidebar.number_input("texture_mean")
    perimeter_mean= st.sidebar.number_input("perimeter_mean")
    area_mean= st.sidebar.number_input("area_mean")
    smoothness_mean= st.sidebar.number_input("smoothness_mean")
    compactness_mean= st.sidebar.number_input("compactness_mean")
    concavity_mean= st.sidebar.number_input("concavity_mean")
    concave_points_mean= st.sidebar.number_input("concave points_mean")
    symmetry_mean= st.sidebar.number_input("symmetry_mean")
    fractal_dimension_mean= st.sidebar.number_input("fractal_dimension_mean")
    radius_se= st.sidebar.number_input("radius_se")
    texture_se= st.sidebar.number_input("texture_se")
    perimeter_se= st.sidebar.number_input("perimeter_se")
    area_se= st.sidebar.number_input("area_se")
    smoothness_se= st.sidebar.number_input("smoothness_se")
    compactness_se= st.sidebar.number_input("compactness_se")
    concavity_se= st.sidebar.number_input("concavity_se")
    concave_points_se= st.sidebar.number_input("concave points_se")
    symmetry_se= st.sidebar.number_input("symmetry_se")
    fractal_dimension_se= st.sidebar.number_input("fractal_dimension_se")
    radius_worst= st.sidebar.number_input("radius_worst")
    texture_worst= st.sidebar.number_input("texture_worst")
    perimeter_worst= st.sidebar.number_input("perimeter_worst")
    area_worst= st.sidebar.number_input("area_worst")
    smoothness_worst= st.sidebar.number_input("smoothness_worst")
    compactness_worst= st.sidebar.number_input("compactness_worst")
    concavity_worst= st.sidebar.number_input("concavity_worst")
    concave_points_worst= st.sidebar.number_input("concave points_worst")
    symmetry_worst= st.sidebar.number_input("symmetry_worst")
    fractal_dimension_worst= st.sidebar.number_input("fractal_dimension_worst")
    data ={
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concave points_mean': concave_points_mean,
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean,
        'radius_se': radius_se,
        'texture_se': texture_se,
        'perimeter_se': perimeter_se,
        'area_se': area_se,
        'smoothness_se': smoothness_se,
        'compactness_se': compactness_se,
        'concavity_se': concavity_se,
        'concave points_se': concave_points_se,
        'symmetry_se': symmetry_se,
        'fractal_dimension_se': fractal_dimension_se,
        'radius_worst': radius_worst,
        'texture_worst': texture_worst,
        'perimeter_worst': perimeter_worst,
        'area_worst': area_worst,
        'smoothness_worst': smoothness_worst,
        'compactness_worst': compactness_worst,
        'concavity_worst': concavity_worst,
        'concave points_worst': concave_points_worst,
        'symmetry_worst': symmetry_worst,
        'fractal_dimension_worst': fractal_dimension_worst
    }
    personne = pd.DataFrame(data, index=[0])
    return personne
df2=user_input()
if st.sidebar.button("Envoyer les données"):
    svm_pred= svm.predict(df2)
    st.subheader('Données de patient:')
    st.write(df2)
    if svm_pred == 1:
        st.subheader("Le patient est Maligne.")
    else: st.subheader("Le patient est Bénigne,")