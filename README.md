# 💰 Interface de Prédiction de Salaires avec Machine Learning

Une application Streamlit pour prédire les salaires en utilisant plusieurs modèles de Machine Learning.

## 📋 Caractéristiques

- **3 Datasets disponibles :**
  - Kaggle Software Industry Salary (Salaires IT 2022)
  - Kaggle Salary Data (Données générales de salaires)
  - NBA Salary Dataset (Salaires des joueurs NBA)

- **6 Modèles de ML :**
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting

- **Fonctionnalités :**
  - Comparaison des performances des modèles
  - Visualisation des prédictions vs réalité
  - Interface interactive pour faire des prédictions
  - Métriques complètes (MAE, MSE, RMSE, R²)

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Étapes d'installation

1. **Cloner ou accéder au dossier du projet :**
```bash
cd c:\Users\Tliba\Documents\Esprit\ML\ datawarehouse\interface
```

2. **Créer un environnement virtuel (optionnel mais recommandé) :**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Installer les dépendances :**
```bash
pip install -r requirements.txt
```

## 📖 Utilisation

### Lancer l'application

```bash
streamlit run streamlit_app.py
```

L'application s'ouvrira dans votre navigateur par défaut à `http://localhost:8501`

### Guide d'utilisation

1. **Sélectionner un Dataset :**
   - Utilisez le menu déroulant dans la barre latérale pour choisir le dataset

2. **Analyser les Performances :**
   - Visualisez les métriques de tous les modèles
   - Consultez le meilleur modèle selon R² Score
   - Explorez les graphiques de prédictions vs réalité

3. **Faire une Prédiction :**
   - Entrez les valeurs pour chaque feature
   - Cliquez sur "Faire une Prédiction"
   - Consultez les prédictions de tous les modèles

## 📊 Structure des Fichiers

```
interface/
├── streamlit_app.py              # Application principale
├── requirements.txt              # Dépendances Python
├── kaggle Software Industry Salary Dataset.ipynb
├── kaggle_Salary_data.ipynb
└── NBA_salary_dataset.ipynb
```

## 🔧 Configuration

### Variables d'environnement requises (optionnel)
- Les données sont chargées depuis Kaggle Hub et HuggingFace Datasets

### Authentification Kaggle
Si vous rencontrez des problèmes d'accès aux datasets Kaggle :
1. Visitez https://www.kaggle.com/settings/account
2. Cliquez sur "Create New API Token"
3. Placez le fichier `kaggle.json` dans `C:\Users\<YourUsername>\.kaggle\`

## 📈 Métriques Expliquées

- **MAE (Mean Absolute Error)** : Erreur moyenne absolue - plus bas est mieux
- **MSE (Mean Squared Error)** : Erreur quadratique moyenne - plus bas est mieux
- **RMSE (Root Mean Squared Error)** : Racine de MSE - même unité que la variable cible
- **R² Score** : Coefficient de détermination (0-1) - plus haut est mieux

## ⚠️ Notes Importantes

- L'application utilise la normalisation des données pour le dataset NBA
- Les données manquantes sont complétées par la moyenne (numériques) ou le mode (catégoriques)
- Les variables catégorielles sont automatiquement encodées
- Le train/test split est fixé à 80/20

## 🐛 Résolution des Problèmes

### Erreur "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Erreur de connexion Kaggle
- Vérifiez votre connexion internet
- Configurez votre authentification Kaggle
- Essayez de télécharger le dataset manuellement

### Application lente
- Réduisez la taille du dataset
- Diminuez le nombre de estimateurs dans Random Forest

## 📝 Licence

Ce projet est fourni à titre éducatif.

## 👥 Auteur

Créé pour l'analyse de données et la prédiction de salaires.
