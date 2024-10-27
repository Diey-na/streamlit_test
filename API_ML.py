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
from PIL import Image
import base64
from io import BytesIO

# Configuration de l'application
st.set_page_config(
    page_title="Application de machine learning",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# Chargement de l'icône/image
image = Image.open("Photo d'Identite.jpg")  # Remplacez par le chemin de votre image

# Création de colonnes avec une petite colonne pour l'icône
col_icon, col_content = st.columns([0.2, 4])  # La première colonne est étroite pour l'icône

# Affichage de l'image comme icône dans la colonne de gauche
with col_icon:
    st.image(image, width=100)  
with st.container():
    st.write("**Ingenieure Data Scientist**")  # Titre sous l'image

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


# Ajouter du CSS pour colorer les boutons et définir une image de fond


def get_base64_of_image(image_path):
    """Convertit une image en base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def display_background():
    # Convertissez l'image en base64
    image_base64 = get_base64_of_image("images.png")
    
    # Appliquez l'image en tant que fond via CSS
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url("data:image/png;base64,{image_base64}");
                background-attachment: fixed;
                background-size: cover;
                background-position: center;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Utilisation de la fonction pour afficher le fond
display_background()


# Fonction pour charger les données
def charger_donnees(uploaded_file, file_type):
    """Charge le fichier selon son type."""
    if file_type == "CSV":
        return pd.read_csv(uploaded_file)
    elif file_type == "Texte (txt)":
        return pd.read_csv(uploaded_file, delimiter='\t')
    elif file_type == "JSON":
        return pd.read_json(uploaded_file)
    elif file_type in ["Excel (xlsx)", "Excel (xls)"]:
        return pd.read_excel(uploaded_file)
    elif file_type == "Pickle":
        return pd.read_pickle(uploaded_file)
    elif file_type == "Parquet":
        return pd.read_parquet(uploaded_file)
    else:
        st.error("Type de fichier non supporté.")
        return None

# Fonction pour afficher les informations sur les colonnes et les valeurs manquantes
# Fonction pour afficher les types de données et les valeurs manquantes
def afficher_infos_colonnes(data):
    """Affiche les types de données et les valeurs manquantes."""
    st.subheader("Valeurs manquantes")
    
    # Afficher les types de données pour chaque colonne
    # dtypes = data.dtypes
    # st.write("Types de données pour chaque colonne :")
    # st.write(dtypes)

    # Identifier les colonnes avec des valeurs manquantes
    missing_data = data.isnull().sum()
    missing_columns = missing_data[missing_data > 0]

    # Afficher le nombre de valeurs manquantes
    st.write("Nombre de valeurs manquantes pour chaque colonne :")
    st.write(missing_data)

    if not missing_columns.empty:
        st.write("Colonnes ayant des valeurs manquantes :")
        st.write(missing_columns)
    else:
        st.write("Aucune colonne ne contient de valeurs manquantes.")


# Fonction pour traiter les valeurs manquantes
def traiter_valeurs_manquantes(data, column_to_treat, treatment_option):
    """Traite les valeurs manquantes selon l'option choisie."""
    if treatment_option == "Moyenne":
        mean_value = data[column_to_treat].mean()
        data[column_to_treat].fillna(mean_value, inplace=True)
        st.write(f"Valeurs manquantes dans {column_to_treat} traitées par la moyenne : {mean_value:.2f}.")
    
    elif treatment_option == "Médiane":
        median_value = data[column_to_treat].median()
        data[column_to_treat].fillna(median_value, inplace=True)
        st.write(f"Valeurs manquantes dans {column_to_treat} traitées par la médiane : {median_value:.2f}.")
    
    elif treatment_option == "Mode":
        mode_value = data[column_to_treat].mode()[0]
        data[column_to_treat].fillna(mode_value, inplace=True)
        st.write(f"Valeurs manquantes dans {column_to_treat} traitées par le mode : {mode_value}.")
    
    elif treatment_option == "Supprimer la colonne":
        data.drop(columns=[column_to_treat], inplace=True)
        st.write(f"La colonne {column_to_treat} a été supprimée.")
    
    elif treatment_option == "Supprimer les lignes":
        data.dropna(subset=[column_to_treat], inplace=True)
        st.write(f"Les lignes contenant des valeurs manquantes dans la colonne {column_to_treat} ont été supprimées.")
    
    elif treatment_option == "Validation croisée":
        st.write(f"Pour la colonne {column_to_treat}, un traitement par validation croisée est recommandé pour une analyse plus approfondie.")
    
    st.write("Données après traitement des valeurs manquantes :")
    st.write(data)
    
    return data

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

# Fonction pour afficher les distributions des colonnes numériques
def afficher_distribution_numeriques(data, column):
    plt.figure(figsize=(10, 5))
    sns.histplot(data[column], kde=True, bins=30)
    plt.title(f'Distribution de la colonne {column}')
    plt.xlabel(column)
    plt.ylabel('Fréquence')
    plt.grid()
    st.pyplot(plt)
    plt.clf()  # Clear the figure after displaying to avoid overlaps

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

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # K-Fold validation
    kf = KFold(n_splits=k_folds)

    # Sélection du modèle
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
    else:
        st.error("Modèle non reconnu.")
        return

    # Entraîner le modèle avec validation croisée K-Fold
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy' if model_type.endswith('Classification') else 'neg_mean_squared_error')

    avg_score = scores.mean()
    if model_type.endswith("Classification"):
        st.write(f"Score de précision moyen avec {k_folds} folds: {avg_score:.2f}")
    else:
        st.write(f"Erreur quadratique moyenne avec {k_folds} folds: {-avg_score:.2f}")

    # Entraîner le modèle sur l'ensemble de données d'entraînement complet
    model.fit(X_train, y_train)
    st.write(f"Modèle {model_type} entraîné avec succès.")

    # Faire des prédictions sur l'ensemble de test
    predictions = model.predict(X_test)

    # Évaluer les performances du modèle
    if model_type.endswith("Classification"):
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Précision sur l'ensemble de test : {accuracy:.2f}")
    else:
        mse = mean_squared_error(y_test, predictions)
        st.write(f"Erreur quadratique moyenne sur l'ensemble de test : {mse:.2f}")

    # Sauvegarder les résultats dans un DataFrame
    results = pd.DataFrame(data={"Réel": y_test, "Prédiction": predictions})
    st.session_state.results = results

# Interface de l'application
def main():
    st.title('Importation des données')
    
    # Sélection du type de fichier
    file_type = st.selectbox(
        "Sélectionnez le type de fichier à charger :",
        ("CSV", "Texte (txt)", "JSON", "Excel (xlsx)", "Pickle", "Parquet")
    )
    
    # Téléchargement du fichier
    uploaded_file = st.file_uploader("Téléchargez votre fichier", type=['csv', 'txt', 'json', 'xlsx', 'xls', 'pkl', 'parquet'])
    
    # Chargement des données
    if st.button("Charger les données", key="load_data"):
        try:
            st.session_state.data = charger_donnees(uploaded_file, file_type)
            if st.session_state.data is not None:
                st.write("Données chargées avec succès ! Voici un aperçu :")
                st.write(st.session_state.data.head())
                st.session_state.numeric_columns = st.session_state.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
                afficher_infos_colonnes(st.session_state.data)
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")


    # Traitement si les données ont été chargées
    if "data" in st.session_state:
        st.write("Aperçu des données :")
        st.write(st.session_state.data.head())

        # Afficher les informations sur les colonnes
        afficher_infos_colonnes(st.session_state.data)


        # Matrice de corrélation
        # if st.button("Afficher la matrice de corrélation"):
        #     afficher_matrice_correlation(st.session_state.data)

                # Afficher la matrice de corrélation
        afficher_matrice_correlation(st.session_state.data)

        # Sélectionner la colonne pour afficher la distribution
        st.subheader("Distribution des colonnes")
        all_columns = st.session_state.data.columns.tolist()
        column_to_plot = st.selectbox("Choisissez une colonne à visualiser :", all_columns)
        if column_to_plot and pd.api.types.is_numeric_dtype(st.session_state.data[column_to_plot]):
            afficher_distribution_numeriques(st.session_state.data, column_to_plot)

        # Traitement des valeurs manquantes
        st.subheader("Traitement des valeurs manquantes")
        missing_values = st.session_state.data.isnull().sum()
        columns_with_nan = missing_values[missing_values > 0].index.tolist()
        
        if columns_with_nan:
            for column in columns_with_nan:
                st.write(f"Colonne : {column}")
                treatment_option = st.selectbox(
                    f"Choisissez la méthode de traitement pour {column} :",
                    ["Moyenne", "Médiane", "Mode", "Supprimer la colonne", "Supprimer les lignes","Validation croisée"],
                    key=column
                )
                if st.button(f"Appliquer le traitement pour {column}", key=f"apply_{column}"):
                    st.session_state.data = traiter_valeurs_manquantes(st.session_state.data, column, treatment_option)
                    st.write(f"Nombre de valeurs manquantes pour {column} après traitement :")
                    st.write(st.session_state.data[column].isnull().sum())
        else:
            st.write("Aucune colonne avec des valeurs manquantes.")
        
        # Encodage des variables
        st.subheader("Encodage des Variables")
        categorical_columns = st.selectbox("Choisissez une colonne catégorielle à encoder :", st.session_state.data.select_dtypes(include=['object']).columns.tolist())
        encoding_type = st.selectbox("Choisissez le type d'encodage :", ["Label Encoding", "One-Hot Encoding"])
        if st.button("Encoder", key="encode"):
            st.session_state.data = encoder_variables(st.session_state.data, encoding_type, categorical_columns)

        # Entraînement de modèle
        st.subheader("Entraînement de Modèle")
        
        target_variable = st.selectbox("Choisissez la variable cible :", st.session_state.data.columns.tolist())
        
        explanatory_variables = st.multiselect("Choisissez les variables explicatives :", 
                                                st.session_state.data.columns.tolist(), 
                                                default=st.session_state.data.columns.tolist()[:-1])

        # Ajouter K-Folds et K pour les K-plus proches voisins
        k_folds = st.number_input("Nombre de folds pour K-Fold (validation croisée)", min_value=2, max_value=20, value=5)
        n_neighbors = st.number_input("Nombre de voisins pour KNN", min_value=1, max_value=20, value=5)

        model_type = st.selectbox("Choisissez le type de modèle :", 
                                   ["Régression Logistique", "Arbre de Décision (Classification)", "SVM (Classification)", 
                                    "KNN (Classification)", "Arbre de Décision (Régression)", "Régression Linéaire", 
                                    "SVM (Régression)", "KNN (Régression)"])

        # Bouton pour entraîner le modèle
        if st.button("Entraîner le modèle", key="train_model"):
            entrainer_model(st.session_state.data, model_type, target_variable, explanatory_variables, k_folds, n_neighbors)
        
        # Afficher les résultats de prédiction
        if "results" in st.session_state:
            st.subheader("Résultats de Prédiction")
            st.write(st.session_state.results)
            
            csv = st.session_state.results.to_csv(index=False).encode('utf-8')
            st.download_button(label="Télécharger les résultats en CSV", data=csv, file_name='predictions.csv', mime='text/csv')
    else:
        st.write("Veuillez télécharger un fichier pour commencer.")

# Exécution de l'application
if __name__ == "__main__":
    st.title('Application de machine learning')
    main()
    
