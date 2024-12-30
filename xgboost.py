import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Charger les données préparées
file_path = 'dataset_prep.csv'
data = pd.read_csv(file_path)

# Réduire la taille du dataset (20% des données)
data = data.sample(frac=0.2, random_state=42)

# Encoder la colonne 'code_plus'
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

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionnaire pour stocker les modèles pour chaque colonne cible
models = {}
reports = {}

# Entraîner un modèle distinct pour chaque colonne cible
for col in colonnes_cibles:
    print(f"Entraînement du modèle pour la colonne cible : {col}")
    
    # Récupérer toutes les classes possibles dans le dataset complet
    all_classes = data[col].unique()
    
    # Vérifier et réindexer les cibles pour inclure toutes les classes
    y_train_reindexed = y_train[col].apply(lambda x: x if x in all_classes else -1)
    y_test_reindexed = y_test[col].apply(lambda x: x if x in all_classes else -1)

    # Définir le modèle avec toutes les classes possibles
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    
    # Entraîner le modèle
    models[col] = xgb.fit(X_train, y_train_reindexed)
    
    # Prédictions sur l'ensemble de test
    y_pred = xgb.predict(X_test)
    
    # Générer un rapport de classification
    report = classification_report(y_test[col], y_pred, zero_division=0)
    reports[col] = report
    print(f"Rapport pour {col} :\n{report}\n")

# Sauvegarder tous les modèles
for col, model in models.items():
    model_path = f'modele_xgboost_{col}.pkl'
    joblib.dump(model, model_path)
    print(f"Modèle pour '{col}' sauvegardé sous : {model_path}")

# Sauvegarder les rapports dans un fichier
with open("classification_reports.txt", "w") as f:
    for col, report in reports.items():
        f.write(f"Rapport pour {col} :\n{report}\n\n")

print("Tous les rapports de classification sauvegardés dans 'classification_reports.txt'.")
