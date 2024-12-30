import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib
from sklearn.preprocessing import LabelEncoder

# Charger les données préparées
file_path = 'dataset_prep.csv'
data = pd.read_csv(file_path)

# Réduire la taille du dataset (20% des données)
data = data.sample(frac=0.2, random_state=42)

# Encoder la colonne code_plus
label_encoder = LabelEncoder()
data['code_plus'] = label_encoder.fit_transform(data['code_plus'])

# Séparer les variables d'entrée et les cibles
colonnes_entrees = [
    'year', 'month', 'day', 'hour', 'weekday', 'PREM_TYP_DESC',
    'IN_PARK', 'IN_PUBLIC_HOUSING', 'IN_STATION', 'VIC_AGE_GROUP',
    'VIC_RACE', 'VIC_SEX', 'code_plus'
]
colonnes_cibles = ['OFNS_DESC', 'SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX']

X = data[colonnes_entrees]
y = data[colonnes_cibles]

# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle multi-sorties avec des paramètres optimisés
print("Entraînement du modèle...")
rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
model = MultiOutputClassifier(rf)
model.fit(X_train, y_train)

# Évaluer le modèle
print("Évaluation du modèle...")
y_pred = model.predict(X_test)

# Générer un rapport de classification pour chaque cible
for i, cible in enumerate(colonnes_cibles):
    print(f"Rapport pour {cible} :")
    print(classification_report(y_test.iloc[:, i], [pred[i] for pred in y_pred]))

# Sauvegarder le modèle
output_model_path = 'modele_multi_sorties.pkl'
joblib.dump(model, output_model_path)
print(f"Modèle sauvegardé sous : {output_model_path}")
