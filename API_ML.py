import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Configuration de l'application
st.set_page_config(
    page_title="Application de machine learning",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# Ajouter du CSS pour colorer les boutons
st.markdown("""
<style>
    .stButton > button {
        background-color: green;
        color: white;
        border: none;
        padding: 10px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger les données
def charger_donnees(uploaded_file, file_type):
    """Charge le fichier selon son type et gère l'erreur pyarrow."""
    try:
        if file_type == "CSV":
            data = pd.read_csv(uploaded_file)
        elif file_type == "Texte (txt)":
            data = pd.read_csv(uploaded_file, delimiter='\t')
        elif file_type == "JSON":
            data = pd.read_json(uploaded_file)
        elif file_type in ["Excel (xlsx)", "Excel (xls)"]:
            data = pd.read_excel(uploaded_file)
        elif file_type == "Pickle":
            data = pd.read_pickle(uploaded_file)
        elif file_type == "Parquet":
            data = pd.read_parquet(uploaded_file)
        else:
            st.error("Type de fichier non supporté.")
            return None

        return data

    except Exception as e:
        # Gérer spécifiquement l'erreur pyarrow/numpy
        st.error(f"Erreur lors du chargement des données : {e}")

        if "numpy.dtype" in str(e):
            st.warning("Problème avec numpy.dtype et pyarrow. Tentative de conversion des colonnes problématiques en chaînes de caractères.")
            # Conversion des colonnes de type object ou category en chaînes de caractères
            data = pd.read_csv(uploaded_file) if file_type == "CSV" else pd.read_excel(uploaded_file)
            object_columns = data.select_dtypes(include=['object', 'category']).columns
            data[object_columns] = data[object_columns].astype(str)
            return data
        else:
            return None

# Fonction pour afficher les informations sur les colonnes
def afficher_infos_colonnes(data):
    """Affiche les types de données et les valeurs manquantes."""
    st.subheader("Types de données et valeurs manquantes")
    dtypes = data.dtypes
    st.write("Types de données pour chaque colonne :")
    st.write(dtypes)

    # Identifier les colonnes avec des valeurs manquantes
    missing_data = data.isnull().sum()
    missing_columns = missing_data[missing_data > 0]

    st.write("Nombre de valeurs manquantes pour chaque colonne :")
    st.write(missing_data)

    if not missing_columns.empty:
        st.write("Colonnes ayant des valeurs manquantes :")
        st.write(missing_columns)
    else:
        st.write("Aucune colonne ne contient de valeurs manquantes.")

# Fonction pour afficher la matrice de corrélation
def afficher_matrice_correlation(data):
    """Affiche la matrice de corrélation et sa heatmap."""
    colonnes_numeriques = data.select_dtypes(include=['float64', 'int64'])

    if colonnes_numeriques.empty:
        st.write("Aucune variable numérique disponible pour calculer la matrice de corrélation.")
        return

    correlation_matrix = colonnes_numeriques.corr()
    st.subheader("Matrice de corrélation")
    st.write(correlation_matrix)

    st.subheader("Carte thermique (Heatmap) de la corrélation")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# Fonction pour encoder les variables
def encoder_variables(data, encoding_type, column):
    """Encode la colonne selon le type d'encodage choisi."""
    if encoding_type == "Label Encoding":
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        st.write(f"Colonne {column} encodée avec Label Encoding.")
    elif encoding_type == "One-Hot Encoding":
        data = pd.get_dummies(data, columns=[column], drop_first=True)
        st.write(f"Colonne {column} encodée avec One-Hot Encoding.")
    return data

# Fonction pour entraîner un modèle
def entrainer_model(data, model_type, target_variable, explanatory_variables, k_folds, n_neighbors):
    """Entraîne un modèle de machine learning selon le type sélectionné."""
    X = data[explanatory_variables]
    y = data[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    kf = KFold(n_splits=k_folds)

    model = None
    if model_type == "Régression Logistique":
        model = LogisticRegression()
    elif model_type == "Arbre de Décision (Classification)":
        model = DecisionTreeClassifier()
    elif model_type == "SVM (Classification)":
        model = SVC()
    elif model_type == "KNN (Classification)":
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_type == "Arbre de Décision (Régression)":
        model = DecisionTreeRegressor()
    elif model_type == "Régression Linéaire":
        model = LinearRegression()
    elif model_type == "SVM (Régression)":
        model = SVR()
    elif model_type == "KNN (Régression)":
        model = KNeighborsRegressor(n_neighbors=n_neighbors)

    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy' if model_type.endswith('Classification') else 'neg_mean_squared_error')

    avg_score = scores.mean()
    if model_type.endswith("Classification"):
        st.write(f"Score de précision moyen avec {k_folds} folds: {avg_score:.2f}")
    else:
        st.write(f"Erreur quadratique moyenne avec {k_folds} folds: {-avg_score:.2f}")

    model.fit(X_train, y_train)
    st.write(f"Modèle {model_type} entraîné avec succès.")

    predictions = model.predict(X_test)
    if model_type.endswith("Classification"):
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Précision sur l'ensemble de test : {accuracy:.2f}")
    else:
        mse = mean_squared_error(y_test, predictions)
        st.write(f"Erreur quadratique moyenne sur l'ensemble de test : {mse:.2f}")

    results = pd.DataFrame(data={"Réel": y_test, "Prédiction": predictions})
    st.session_state.results = results

# Interface de l'application
def main():
    st.title('Importation des données')

    file_type = st.selectbox(
        "Sélectionnez le type de fichier à charger :",
        ("CSV", "Texte (txt)", "JSON", "Excel (xlsx)", "Pickle", "Parquet")
    )

    uploaded_file = st.file_uploader("Téléchargez votre fichier", type=['csv', 'txt', 'json', 'xlsx', 'xls', 'pkl', 'parquet'])

    if st.button("Charger les données", key="load_data"):
        try:
            st.session_state.data = charger_donnees(uploaded_file, file_type)
            if st.session_state.data is not None:
                st.write("Données chargées avec succès ! Voici un aperçu :")
                st.write(st.session_state.data.head())
                afficher_infos_colonnes(st.session_state.data)
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")

    if "data" in st.session_state:
        st.write("Aperçu des données :")
        st.write(st.session_state.data.head())

        afficher_infos_colonnes(st.session_state.data)

        if st.button("Afficher la matrice de corrélation"):
            afficher_matrice_correlation(st.session_state.data)

        st.subheader("Encodage des Variables")
        categorical_columns = st.selectbox("Choisissez une colonne catégorielle à encoder :", st.session_state.data.select_dtypes(include=['object']).columns.tolist())
        encoding_type = st.selectbox("Choisissez le type d'encodage :", ["Label Encoding", "One-Hot Encoding"])
        if st.button("Encoder", key="encode"):
            st.session_state.data = encoder_variables(st.session_state.data, encoding_type, categorical_columns)

        st.subheader("Entraînement du Modèle")
        target_variable = st.selectbox("Sélectionnez la variable cible (target) :", st.session_state.data.columns)
        explanatory_variables = st.multiselect("Sélectionnez les variables explicatives (features) :", st.session_state.data.columns)
        model_type = st.selectbox("Choisissez le type de modèle :", ["Régression Logistique", "Arbre de Décision (Classification)", "SVM (Classification)", "KNN (Classification)", "Arbre de Décision (Régression)", "Régression Linéaire", "SVM (Régression)", "KNN (Régression)"])
        k_folds = st.slider("Choisissez le nombre de folds pour la validation croisée :", min_value=2, max_value=10, value=5)
        n_neighbors = st.slider("Choisissez le nombre de voisins (pour KNN) :", min_value=1, max_value=20, value=5)

        if st.button("Entraîner le modèle"):
            entrainer_model(st.session_state.data, model_type, target_variable, explanatory_variables, k_folds, n_neighbors)

if __name__ == "__main__":
    main()
