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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Titre de l'application
st.title('Application de machine learning')

# Ajouter du CSS pour colorer les boutons
st.markdown("""
<style>
    .green-button {
        background-color: green !important;
        color: white !important;
        border: none;
        padding: 10px;
        cursor: pointer;
    }
    .stButton > button {
        background-color: green;
        color: white;
        border: none;
        padding: 10px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger les données en fonction du type de fichier
def charger_donnees(uploaded_file, file_type):
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

# Fonction pour afficher l'analyse descriptive
def afficher_analyse_descriptive(data):
    st.subheader("Analyse descriptive des données")
    
    # Statistiques de base
    st.write("Statistiques de base :")
    st.write(data.describe(include='all'))
    
    # Type des données
    st.write("Types de données pour chaque colonne :")
    st.write(data.dtypes)
    
    # Valeurs manquantes
    st.write("Nombre de valeurs manquantes par colonne :")
    missing_values = data.isnull().sum()
    st.write(missing_values)
    
    return missing_values

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

# Fonction pour traiter les valeurs manquantes
def traiter_valeurs_manquantes(data, column_to_treat, treatment_option):
    if treatment_option == "Moyenne":
        mean_value = data[column_to_treat].mean()
        data[column_to_treat].fillna(mean_value, inplace=True)
        st.write(f"Valeurs manquantes dans {column_to_treat} traitées par la moyenne : {mean_value}.")
    
    elif treatment_option == "Médiane":
        median_value = data[column_to_treat].median()
        data[column_to_treat].fillna(median_value, inplace=True)
        st.write(f"Valeurs manquantes dans {column_to_treat} traitées par la médiane : {median_value}.")
    
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
    
    # Afficher les données mises à jour après le traitement des valeurs manquantes
    st.write("Données après traitement des valeurs manquantes :")
    st.write(data)
    
    return data

# Fonction pour afficher la matrice de corrélation
def afficher_matrice_correlation(data):
    # Filtrer uniquement les colonnes numériques
    colonnes_numeriques = data.select_dtypes(include=['float64', 'int64'])

    if colonnes_numeriques.empty:
        st.write("Aucune variable numérique disponible pour calculer la matrice de corrélation.")
        return

    # Calcul de la matrice de corrélation
    correlation_matrix = colonnes_numeriques.corr()

    # Affichage de la matrice sous forme de tableau
    st.subheader("Matrice de corrélation")
    st.write(correlation_matrix)

    # Affichage de la matrice sous forme de heatmap
    st.subheader("Carte thermique (Heatmap) de la corrélation")
    fig, ax = plt.subplots(figsize=(10, 6))  # Créer une figure et un axe
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
    plt.close(fig)
   
# Fonction pour encoder les variables
def encoder_variables(data, encoding_type, column):
    if encoding_type == "Label Encoding":
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        st.write(f"Colonne {column} encodée avec Label Encoding.")
    elif encoding_type == "One-Hot Encoding":
        data = pd.get_dummies(data, columns=[column], drop_first=True)
        st.write(f"Colonne {column} encodée avec One-Hot Encoding.")
    return data

# Fonction pour décoder les variables
def decoder_variables(data, column):
    # Simple binarisation ou conditionnelle pour démonstration
    conditions = [
        (data[column] < 10),  # par exemple, considérant moins de 10 comme 'Bas'
        (data[column] >= 10)  # et 10 ou plus comme 'Haut'
    ]
    choices = ['Bas', 'Haut']
    data[column] = np.select(conditions, choices)
    st.write(f"Colonne {column} décodée en catégories.")
    return data

# Fonction pour entraîner un modèle
def entrainer_model(data, model_type, target_variable, explanatory_variables, k_folds, n_neighbors):
    X = data[explanatory_variables]
    y = data[target_variable]

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # K-Fold validation
    kf = KFold(n_splits=k_folds)

    # Sélection du modèle
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
        st.write("Modèle non reconnu.")
        return

    # Entraîner le modèle avec validation croisée K-Fold
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy' if model_type.endswith('Classification') else 'neg_mean_squared_error')

    # Calcul de la moyenne des scores
    avg_score = scores.mean()

    # Afficher le score moyen de la validation croisée
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

# Interface de l'application
def main():
    # Téléchargement de fichiers
    st.sidebar.title("Téléchargement de données")
    uploaded_file = st.sidebar.file_uploader("Choisissez un fichier", type=["csv", "txt", "json", "xlsx", "xls", "pickle", "parquet"])
    
    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[-1].upper()
        st.session_state.data = charger_donnees(uploaded_file, file_type)

        st.write("Aperçu des données :")
        st.write(st.session_state.data.head())
        
        st.write("Analyse descriptive des données :")
        missing_values = afficher_analyse_descriptive(st.session_state.data)

        # Traitement des valeurs manquantes
        st.subheader("Traitement des valeurs manquantes")
        column_to_treat = st.selectbox("Choisissez une colonne à traiter :", st.session_state.data.columns)
        treatment_option = st.selectbox("Choisissez une option de traitement :", ["Moyenne", "Médiane", "Mode", "Supprimer la colonne", "Supprimer les lignes", "Validation croisée"])
        
        if st.button("Appliquer traitement"):
            st.session_state.data = traiter_valeurs_manquantes(st.session_state.data, column_to_treat, treatment_option)

        # Afficher la matrice de corrélation
        afficher_matrice_correlation(st.session_state.data)

        # Sélectionner la colonne pour afficher la distribution
        st.subheader("Distribution des colonnes")
        all_columns = st.session_state.data.columns.tolist()
        column_to_plot = st.selectbox("Choisissez une colonne à visualiser :", all_columns)
        if column_to_plot and pd.api.types.is_numeric_dtype(st.session_state.data[column_to_plot]):
            afficher_distribution_numeriques(st.session_state.data, column_to_plot)

        # Encoder les variables
        st.subheader("Encodage des variables")
        column_to_encode = st.selectbox("Choisissez une colonne à encoder :", all_columns)
        encoding_type = st.selectbox("Choisissez le type d'encodage :", ["Label Encoding", "One-Hot Encoding"])
        
        if st.button("Appliquer encodage"):
            st.session_state.data = encoder_variables(st.session_state.data, encoding_type, column_to_encode)

        # Décoder les variables
        st.subheader("Décodage des variables")
        column_to_decode = st.selectbox("Choisissez une colonne à décoder :", all_columns)
        
        if st.button("Appliquer décodage"):
            st.session_state.data = decoder_variables(st.session_state.data, column_to_decode)

        # Entraîner un modèle
        st.subheader("Entraînement du modèle")
        model_type = st.selectbox("Choisissez un modèle :", ["Régression Logistique", "Arbre de Décision (Classification)", "SVM (Classification)", "KNN (Classification)", 
                                                            "Arbre de Décision (Régression)", "Régression Linéaire", "SVM (Régression)", "KNN (Régression)"])
        target_variable = st.selectbox("Choisissez la variable cible :", all_columns)
        explanatory_variables = st.multiselect("Choisissez les variables explicatives :", all_columns, default=[all_columns[0]])
        k_folds = st.number_input("Choisissez le nombre de folds pour la validation croisée :", min_value=2, max_value=10, value=5)
        n_neighbors = st.number_input("Choisissez le nombre de voisins pour KNN :", min_value=1, max_value=20, value=5)

        if st.button("Entraîner le modèle"):
            entrainer_model(st.session_state.data, model_type, target_variable, explanatory_variables, k_folds, n_neighbors)

if __name__ == "__main__":
    main()
