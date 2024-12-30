import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib
from sklearn.preprocessing import LabelEncoder

# Charger les datasets avant et après encodage
file_path_original = 'dataset_avec_code_plus.csv'  # Dataset avant encodage
file_path_encoded = 'dataset_prep.csv'  # Dataset après encodage
data_original = pd.read_csv(file_path_original)
data_encoded = pd.read_csv(file_path_encoded)

# Charger le modèle
model = joblib.load('modele_multi_sorties.pkl')

# Créer des mappings pour chaque colonne catégorique
colonnes_categoriques = [
    'weekday', 'PREM_TYP_DESC', 'VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX',
    'OFNS_DESC', 'SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX'
]
mappings = {}
for col in colonnes_categoriques:
    unique_original = data_original[col].dropna().unique()
    unique_encoded = data_encoded[col].dropna().unique()
    mappings[col] = dict(zip(unique_encoded, unique_original))

st.title("Prédiction des caractéristiques des crimes à New York")

# Carte interactive pour sélectionner une région
st.subheader("Sélectionnez une région sur la carte")
map_center = [40.7128, -74.0060]  # Centre sur New York
crime_map = folium.Map(location=map_center, zoom_start=12)

# Ajouter une interaction pour capturer les coordonnées
crime_map.add_child(folium.LatLngPopup())
map_data = st_folium(crime_map, width=700, height=500)

# Récupérer les coordonnées de la dernière sélection
latitude, longitude = 37.3000, -50.0060  # Valeurs par défaut
if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
    latitude = map_data["last_clicked"]["lat"]
    longitude = map_data["last_clicked"]["lng"]

st.write(f"Latitude sélectionnée : {latitude}")
st.write(f"Longitude sélectionnée : {longitude}")

# Collecter les autres entrées utilisateur
st.subheader("Entrez les détails du crime")
year = st.number_input("Année", min_value=2000, max_value=2025, value=2023)
month = st.number_input("Mois", min_value=1, max_value=12, value=1)
day = st.number_input("Jour", min_value=1, max_value=31, value=1)
hour = st.number_input("Heure", min_value=0, max_value=23, value=12)
weekday = st.selectbox("Jour de la semaine", data_original['weekday'].unique())
prem_typ_desc = st.selectbox("Type de lieu", data_original['PREM_TYP_DESC'].unique())
in_park = st.radio("Dans un parc ?", ["Non", "Oui"])
in_public_housing = st.radio("Dans un logement public ?", ["Non", "Oui"])
in_station = st.radio("Dans une station ?", ["Non", "Oui"])
vic_age_group = st.selectbox("Tranche d'âge de la victime", data_original['VIC_AGE_GROUP'].unique())
vic_race = st.selectbox("Race de la victime", data_original['VIC_RACE'].unique())
vic_sex = st.selectbox("Sexe de la victime", data_original['VIC_SEX'].unique())

# Encoder les entrées utilisateur pour le modèle
input_data = pd.DataFrame([{
    "year": year,
    "month": month,
    "day": day,
    "hour": hour,
    "weekday": data_encoded[data_original['weekday'] == weekday].iloc[0]['weekday'],
    "PREM_TYP_DESC": data_encoded[data_original['PREM_TYP_DESC'] == prem_typ_desc].iloc[0]['PREM_TYP_DESC'],
    "IN_PARK": 1 if in_park == "Oui" else 0,
    "IN_PUBLIC_HOUSING": 1 if in_public_housing == "Oui" else 0,
    "IN_STATION": 1 if in_station == "Oui" else 0,
    "VIC_AGE_GROUP": data_encoded[data_original['VIC_AGE_GROUP'] == vic_age_group].iloc[0]['VIC_AGE_GROUP'],
    "VIC_RACE": data_encoded[data_original['VIC_RACE'] == vic_race].iloc[0]['VIC_RACE'],
    "VIC_SEX": data_encoded[data_original['VIC_SEX'] == vic_sex].iloc[0]['VIC_SEX'],
    "code_plus": int(latitude * longitude)  # Exemple de génération d'un code basé sur les coordonnées
}])

# Prédire avec le modèle
if st.button("Prédire"):
    prediction = model.predict(input_data)
    
    # Décoder les résultats
    decoded_results = {
        "OFNS_DESC": mappings['OFNS_DESC'][prediction[0][0]],
        "SUSP_AGE_GROUP": mappings['SUSP_AGE_GROUP'][prediction[0][1]],
        "SUSP_RACE": mappings['SUSP_RACE'][prediction[0][2]],
        "SUSP_SEX": mappings['SUSP_SEX'][prediction[0][3]]
    }
    
    # Afficher les résultats décodés
    st.subheader("Résultats des prédictions")
    st.write(f"Type de crime (OFNS_DESC) : {decoded_results['OFNS_DESC']}")
    st.write(f"Tranche d'âge du suspect (SUSP_AGE_GROUP) : {decoded_results['SUSP_AGE_GROUP']}")
    st.write(f"Race du suspect (SUSP_RACE) : {decoded_results['SUSP_RACE']}")
    st.write(f"Sexe du suspect (SUSP_SEX) : {decoded_results['SUSP_SEX']}")
