# titanic_ml.py

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1️⃣ Charger le dataset
df = pd.read_csv("train.csv")  # Assure-toi que train.csv est dans le même dossier

# 2️⃣ Supprimer les colonnes inutiles
df = df.drop(columns=['Name', 'Ticket', 'Cabin'])

# 3️⃣ Convertir les colonnes catégorielles en nombres
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# 4️⃣ Définir X et y
X = df.drop(columns=['Survived', 'PassengerId'])
y = df['Survived']

# 5️⃣ Imputer les valeurs manquantes
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 6️⃣ Vérification rapide
print("Vérification des NaN :\n", X.isnull().sum())

# 7️⃣ Séparer train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8️⃣ Créer et entraîner le modèle
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 9️⃣ Prédictions et évaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy sur le test set : {accuracy:.2f}")

#  🔹 Optionnel : prédictions pour les 10 premières lignes
print("\nExemple de prédictions pour 10 premiers passagers :")
for i in range(10):
    print(f"PassengerId {df.iloc[i]['PassengerId']}: prédit={y_pred[i]}, vrai={y_test.iloc[i]}")
