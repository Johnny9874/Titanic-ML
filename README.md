🚢 Titanic Survival Prediction (Machine Learning Project)

📌 Description

Ce projet applique un pipeline Machine Learning supervisé sur le célèbre dataset Titanic afin de prédire la survie des passagers.
Il inclut :

Analyse et nettoyage des données

Gestion des valeurs manquantes avec un imputer

Encodage des variables catégorielles (Sex, Embarked)

Entraînement d’un modèle Logistic Regression avec scikit-learn

Évaluation des performances avec accuracy

📂 Structure du projet

📦 titanic-ml
 ┣ 📜 titanic_ml.py     # Script principal
 ┣ 📜 train.csv         # Dataset d'entraînement
 ┗ 📜 README.md         # Documentation

⚙️ Installation

1. Cloner le projet

git clone https://github.com/<ton-username>/titanic-ml.git
cd titanic-ml

2. Créer un environnement virtuel (optionnel mais recommandé)

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

3. Installer les dépendances

pip install -r requirements.txt

▶️ Utilisation

Exécute simplement le script avec Python :

python titanic_ml.py

Exemple de sortie :
Vérification des NaN :
 Pclass      0
Sex         0
Age         0
SibSp       0
Parch       0
Fare        0
Embarked    0
dtype: int64

Accuracy sur le test set : 0.80

Exemple de prédictions pour 10 premiers passagers :
PassengerId 1.0: prédit=0, vrai=0
PassengerId 2.0: prédit=0, vrai=0
PassengerId 3.0: prédit=0, vrai=1
...

📊 Résultats

Accuracy sur test set : ~80% avec une régression logistique simple

Preuve que certaines features clés (classe du billet, sexe, âge, embarquement) ont une forte influence sur la survie