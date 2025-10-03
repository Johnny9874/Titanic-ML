# titanic_ml.py

''' 
import pandas as pd → importe la librairie pandas (très utilisée pour manipuler les datasets sous forme de DataFrame).
pd est juste un alias pratique pour ne pas taper pandas à chaque fois.
DataFrame = tableau 2D (lignes x colonnes) avec étiquettes, idéal pour stocker tes données Titanic.

from sklearn.impute import SimpleImputer → importe la classe SimpleImputer qui sert à remplacer les valeurs manquantes (NaN) dans le dataset.

from sklearn.model_selection import train_test_split → fonction pour séparer ton dataset en jeu d’entraînement et jeu de test.
Important pour tester ton modèle sur des données qu’il n’a jamais vues.

from sklearn.linear_model import LogisticRegression → importe le modèle de régression logistique.
C’est un modèle ML supervisé pour prédire une variable binaire (ici Survived = 0 ou 1).

from sklearn.metrics import accuracy_score → fonction qui calcule l’accuracy (précision) d’un modèle : nombre de prédictions correctes ÷ nombre total de prédictions.
'''

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1️⃣ Charger le dataset

'''
pd.read_csv() → fonction pandas pour lire un fichier CSV et le convertir en DataFrame.

df = variable qui contient ton dataset complet Titanic.

Chaque ligne = un passager, chaque colonne = une feature (Age, Sex, Survived…).
'''

df = pd.read_csv("train.csv")  # Assure-toi que train.csv est dans le même dossier

# 2️⃣ Supprimer les colonnes inutiles

"""
df.drop() → méthode pandas pour supprimer des colonnes ou lignes.

columns=[...] → indique que l’on supprime des colonnes.

Ici, on enlève des colonnes inutiles pour la prédiction :

Name → trop spécifique, pas de valeur prédictive

Ticket → identifiant du ticket, pas utile

Cabin → trop de valeurs manquantes
"""

df = df.drop(columns=['Name', 'Ticket', 'Cabin'])

# 3️⃣ Convertir les colonnes catégorielles en nombres

"""
df['Sex'] → sélectionne la colonne Sex du DataFrame.

.map({...}) → méthode pandas qui remplace chaque valeur par une valeur correspondante dans un dictionnaire.

Exemple : 'male' → 0, 'female' → 1

On fait pareil pour Embarked (port d’embarquement) : C=0, Q=1, S=2

Pourquoi ? Les modèles ML ne comprennent pas le texte, il faut des nombres.
"""

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# 4️⃣ Définir X et y

"""
X = features / variables d’entrée du modèle

On enlève Survived (la target) et PassengerId (identifiant inutile)

y = target / variable à prédire

Ici, y = la colonne Survived (0 ou 1)

Concept ML : le modèle apprend à prédire y à partir de X.
"""

X = df.drop(columns=['Survived', 'PassengerId'])
y = df['Survived']

# 5️⃣ Imputer les valeurs manquantes

"""
SimpleImputer(strategy='median') → crée un objet imputer qui va remplacer les NaN par la médiane de chaque colonne.

fit_transform(X) → deux actions en une :

fit → calcule la médiane de chaque colonne numérique

transform → remplace chaque NaN par la médiane correspondante

pd.DataFrame(..., columns=X.columns) → on recrée un DataFrame propre avec les mêmes noms de colonnes.
"""

imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 6️⃣ Vérification rapide

"""
X.isnull() → renvoie un DataFrame avec True/False pour chaque cellule si NaN ou non.

.sum() → additionne les True → nombre de NaN par colonne

Sert à s’assurer qu’il n’y a plus de NaN avant d’entraîner le modèle.
"""

print("Vérification des NaN :\n", X.isnull().sum())

# 7️⃣ Séparer train/test
"""
train_test_split() → divise les données en :

X_train / y_train → pour entraîner le modèle

X_test / y_test → pour tester la performance sur des données jamais vues

Paramètres :

test_size=0.2 → 20% des données pour le test

random_state=42 → graine pour reproductibilité

stratify=y → conserve la proportion de survivants/non-survivants dans les deux sets
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8️⃣ Créer et entraîner le modèle

"""
LogisticRegression(max_iter=200) → crée le modèle de régression logistique

max_iter=200 → nombre maximum d’itérations pour que l’algorithme converge

model.fit(X_train, y_train) → apprend les coefficients qui relient chaque feature de X à la probabilité de survie dans y.
"""

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 9️⃣ Prédictions et évaluation

"""
model.predict(X_test) → prédit 0 ou 1 pour chaque passager du test set

accuracy_score(y_test, y_pred) → calcule la proportion de prédictions correctes

f"Accuracy ... {accuracy:.2f}" → f-string pour afficher un float avec 2 décimales
"""

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy sur le test set : {accuracy:.2f}")

#  🔹 Optionnel : prédictions pour les 10 premières lignes

"""
df.iloc[i] → sélectionne la i‑ème ligne du DataFrame

df.iloc[i]['PassengerId'] → récupère l’ID du passager

y_pred[i] → prédiction pour ce passager

y_test.iloc[i] → vraie valeur pour ce passager

La boucle affiche un exemple concret de prédiction vs vérité
"""

print("\nExemple de prédictions pour 10 premiers passagers :")
for i in range(10):
    print(f"PassengerId {df.iloc[i]['PassengerId']}: prédit={y_pred[i]}, vrai={y_test.iloc[i]}")