import joblib
import streamlit as st
import pandas as pd
import joblib

def load_model():
    model = joblib.load('svm_mnist_model.pkl')
    return model

model = load_model()

st.write('''
# Classification du Diagnostic du Cancer du Sein à l'Aide de Modèles de Machine Learning
Ce projet vise à développer un système de classification des patients en fonction de leur diagnostic
médical à partir de données médicales anonymisées. Permettant de prédire si une tumeur mammaire est
bénigne ou maligne en utilisant des techniques de machine learning. Les données utilisées pour former
et évaluer les modèles comprendront des mesures cliniques des cellules présentes dans les biopsies de
tumeurs.
''')


st.sidebar.header("Les parametres d'entrée :")

def user_input():
    radius_worst= st.sidebar.number_input("radius_worst")
    area_worst= st.sidebar.number_input("area_worst")
    concave_points_mean= st.sidebar.number_input("concave points_mean")
    texture_worst= st.sidebar.number_input("texture_worst")
    data ={
        'radius_worst': radius_worst,
        'area_worst': area_worst,
        'concave points_mean': concave_points_mean,
        'texture_worst': texture_worst,
    }
    personne = pd.DataFrame(data, index=[0])
    return personne
df2=user_input()
if st.sidebar.button("Envoyer les données"):
    svm_pred= model.predict(df2)
    st.subheader('Données de patient:')
    st.write(df2)
    if svm_pred == 1:
        st.subheader("Le patient est Maligne.")
    else: st.subheader("Le patient est Bénigne,")