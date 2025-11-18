import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Salary Prediction Interface",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("💰 Interface de Prédiction de Salaires avec Machine Learning")
st.markdown("---")

# Sidebar pour la sélection du dataset
st.sidebar.header("⚙️ Configuration")
dataset_choice = st.sidebar.selectbox(
    "Sélectionnez un dataset:",
    [
        "Kaggle Software Industry Salary",
        "Kaggle Salary Data",
        "NBA Salary Dataset"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    Cette application utilise plusieurs modèles de ML pour prédire les salaires :
    - Linear Regression
    - Decision Tree
    - Ridge Regression
    - Lasso Regression
    - Random Forest
    - Gradient Boosting
    """
)

# Fonctions utilitaires
@st.cache_resource
def load_and_prepare_data(dataset_name):
    """Charge et prépare les données selon le dataset choisi"""
    try:
        if dataset_name == "Kaggle Software Industry Salary":
            import kagglehub
            path = kagglehub.dataset_download("iamsouravbanerjee/software-professional-salaries-2022")
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            csv_path = os.path.join(path, csv_files[0])
            df = pd.read_csv(csv_path)
            target_col = 'Salary'
            
        elif dataset_name == "Kaggle Salary Data":
            import kagglehub
            path = kagglehub.dataset_download("ayeshasiddiqa123/salary-data")
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            csv_path = os.path.join(path, csv_files[0])
            df = pd.read_csv(csv_path)
            target_col = 'Salary'
            
        else:  # NBA Salary Dataset
            df = pd.read_csv("hf://datasets/yvonne90190/NBA_salary_advanced_stats/NBA_Data.csv")
            target_col = 'salary'
        
        return df, target_col
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None, None

def prepare_data(df, target_col):
    """Prépare les données pour l'entraînement"""
    # Gestion des valeurs manquantes
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        else:
            df[col] = df[col].fillna(df[col].mean())
    
    # Encodage des variables catégorielles
    label_encoders = {}
    original_values = {}  # Stocker les valeurs originales
    df_encoded = df.copy()
    
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            original_values[col] = df_encoded[col].unique().tolist()  # Sauvegarder les valeurs originales
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
    
    return df_encoded, label_encoders, original_values

def train_models(X_train, X_test, y_train, y_test, use_scaling=False):
    """Entraîne tous les modèles et retourne les résultats"""
    if use_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(random_state=42),
        'Lasso Regression': Lasso(random_state=42, max_iter=10000),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    results = {}
    predictions = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
        
        predictions[name] = y_pred
        trained_models[name] = model
    
    return results, predictions, trained_models, scaler if use_scaling else None

def plot_model_comparison(results):
    """Crée des graphiques de comparaison des modèles"""
    metrics_df = pd.DataFrame(results).T
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparaison des Modèles', fontsize=16, fontweight='bold')
    
    # MAE
    axes[0, 0].barh(metrics_df.index, metrics_df['MAE'], color='skyblue')
    axes[0, 0].set_title('Mean Absolute Error (MAE)', fontweight='bold')
    axes[0, 0].set_xlabel('MAE')
    
    # MSE
    axes[0, 1].barh(metrics_df.index, metrics_df['MSE'], color='lightcoral')
    axes[0, 1].set_title('Mean Squared Error (MSE)', fontweight='bold')
    axes[0, 1].set_xlabel('MSE')
    
    # RMSE
    axes[1, 0].barh(metrics_df.index, metrics_df['RMSE'], color='lightgreen')
    axes[1, 0].set_title('Root Mean Squared Error (RMSE)', fontweight='bold')
    axes[1, 0].set_xlabel('RMSE')
    
    # R2
    axes[1, 1].barh(metrics_df.index, metrics_df['R2'], color='gold')
    axes[1, 1].set_title('R² Score', fontweight='bold')
    axes[1, 1].set_xlabel('R²')
    
    plt.tight_layout()
    return fig

def plot_predictions_vs_actual(y_test, predictions):
    """Crée des graphiques prédictions vs réalité"""
    n_models = len(predictions)
    cols = 2
    rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('Prédictions vs Réalité', fontsize=16, fontweight='bold')
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes[idx]
        
        ax.scatter(y_test, y_pred, alpha=0.5, s=20)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax.set_xlabel('Valeur Réelle', fontweight='bold')
        ax.set_ylabel('Valeur Prédite', fontweight='bold')
        ax.set_title(f'{model_name}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Masquer les axes inutilisés
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

# Chargement des données
st.subheader("📊 Chargement et Analyse des Données")

with st.spinner("Chargement des données..."):
    df, target_col = load_and_prepare_data(dataset_choice)

if df is not None:
    # Affichage des informations sur le dataset
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre de lignes", len(df))
    with col2:
        st.metric("Nombre de colonnes", len(df.columns))
    with col3:
        st.metric("Colonnes", ", ".join(df.columns[:3]) + "...")
    
    st.markdown("---")
    
    # Préparation des données
    with st.spinner("Préparation des données..."):
        df_encoded, label_encoders, original_values = prepare_data(df.copy(), target_col)
    
    # Affichage des premiers données
    with st.expander("👀 Voir les premières lignes"):
        st.dataframe(df.head(10))
    
    # Séparation des données
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.markdown("---")
    st.subheader("🤖 Entraînement des Modèles")
    
    # Déterminer si on utilise la normalisation (pour NBA)
    use_scaling = dataset_choice == "NBA Salary Dataset"
    
    with st.spinner("Entraînement des modèles en cours..."):
        results, predictions, trained_models, scaler = train_models(X_train, X_test, y_train, y_test, use_scaling)
    
    st.success("✅ Modèles entraînés avec succès!")
    
    # Affichage des résultats d'entraînement
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Métriques de Performance")
        metrics_df = pd.DataFrame(results).T
        st.dataframe(metrics_df, use_container_width=True)
    
    with col2:
        st.write("### Meilleur Modèle (R² Score)")
        best_model = metrics_df['R2'].idxmax()
        best_r2 = metrics_df['R2'].max()
        st.success(f"**{best_model}** avec R² = {best_r2:.4f}")
    
    # Graphiques
    tab1, tab2 = st.tabs(["Comparaison Modèles", "Prédictions vs Réalité"])
    
    with tab1:
        fig1 = plot_model_comparison(results)
        st.pyplot(fig1)
    
    with tab2:
        fig2 = plot_predictions_vs_actual(y_test, predictions)
        st.pyplot(fig2)
    
    st.markdown("---")
    st.subheader("🎯 Prédiction de Salaire")
    
    # Formulaire de prédiction
    st.write("Entrez les valeurs pour faire une prédiction:")
    
    input_data = {}
    cols = st.columns(2)
    
    for idx, col_name in enumerate(X.columns):
        col = cols[idx % 2]
        
        # Vérifier si c'est une colonne catégoriaque (avant encodage)
        if col_name in original_values and col_name in label_encoders:
            # Afficher les valeurs uniques originales
            unique_original_values = sorted(original_values[col_name])
            selected_value = col.selectbox(
                f"{col_name}",
                unique_original_values,
                key=f"select_{col_name}"
            )
            # Encoder la valeur sélectionnée pour la prédiction
            input_data[col_name] = label_encoders[col_name].transform([selected_value])[0]
        else:
            # Pour les variables numériques
            min_val = float(X[col_name].min())
            max_val = float(X[col_name].max())
            mean_val = float(X[col_name].mean())
            input_data[col_name] = col.number_input(
                f"{col_name}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100 if max_val - min_val > 0 else 0.1,
                key=f"input_{col_name}"
            )
    
    # Bouton de prédiction
    if st.button("🚀 Faire une Prédiction", use_container_width=True):
        # Préparer les données d'entrée
        input_df = pd.DataFrame([input_data])
        
        # Appliquer la normalisation si nécessaire
        if scaler:
            input_df_scaled = scaler.transform(input_df)
        else:
            input_df_scaled = input_df
        
        # Faire les prédictions
        st.write("### Résultats des Prédictions")
        
        prediction_results = {}
        for model_name, model in trained_models.items():
            pred = model.predict(input_df_scaled)[0]
            prediction_results[model_name] = pred
        
        # Affichage des résultats
        pred_df = pd.DataFrame(list(prediction_results.items()), columns=['Modèle', 'Salaire Prédit'])
        pred_df = pred_df.sort_values('Salaire Prédit', ascending=False)
        
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        
        # Graphique des prédictions
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(pred_df)))
        bars = ax.barh(pred_df['Modèle'], pred_df['Salaire Prédit'], color=colors)
        ax.set_xlabel('Salaire Prédit', fontweight='bold')
        ax.set_title('Comparaison des Prédictions par Modèle', fontweight='bold', fontsize=14)
        
        # Ajouter les valeurs sur les barres
        for i, (bar, val) in enumerate(zip(bars, pred_df['Salaire Prédit'])):
            ax.text(val, i, f' ${val:.2f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistiques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Moyenne des Prédictions", f"${pred_df['Salaire Prédit'].mean():.2f}")
        with col2:
            st.metric("Prédiction Minimale", f"${pred_df['Salaire Prédit'].min():.2f}")
        with col3:
            st.metric("Prédiction Maximale", f"${pred_df['Salaire Prédit'].max():.2f}")

else:
    st.error("❌ Impossible de charger les données. Veuillez vérifier votre connexion internet.")
