
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
def afficher_distribution_numeriques(data, column_to_plot):
    st.subheader(f"Distribution de la colonne {column_to_plot}")
    
    # Diagramme en barres
    st.write("Diagramme en barres :")
    fig, ax = plt.subplots()
    sns.histplot(data[column_to_plot].dropna(), kde=False, ax=ax)
    st.pyplot(fig)
    
    # Diagramme circulaire (si c'est une variable catégorielle ou si peu de valeurs uniques)
    if data[column_to_plot].nunique() < 10:  # Limité aux colonnes avec moins de 10 valeurs uniques
        st.write("Diagramme circulaire :")
        fig, ax = plt.subplots()
        data[column_to_plot].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)
    else:
        st.write("Le diagramme circulaire n'est pas pertinent pour cette colonne.")
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
    #st.write("Aperçu des données avant le calcul de la matrice de corrélation :")
    #st.write(data.head()) 
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

    # Sauvegarder les prédictions dans un DataFrame
    results = pd.DataFrame(data={"Réel": y_test, "Prédiction": predictions})
    st.session_state.results = results


# Interface de l'application
def main():
    st.title('Importation des données')
    
    # Liste déroulante pour choisir le type de fichier
    file_type = st.selectbox(
        "Sélectionnez le type de fichier à charger :",
        ("CSV", "Texte (txt)", "JSON", "Excel (xlsx)", "Pickle", "Parquet")
    )
    
    # Téléchargement du fichier
    uploaded_file = st.file_uploader("Téléchargez votre fichier", type=['csv', 'txt', 'json', 'xlsx', 'xls', 'pkl', 'parquet'])
    
    # Bouton pour charger les données
    if st.button("Charger les données", key="load_data"):
        try:
            st.session_state.data = charger_donnees(uploaded_file, file_type)
            st.write("Données chargées avec succès ! Voici un aperçu :")
            st.write(st.session_state.data.head())
            st.session_state.missing_values = afficher_analyse_descriptive(st.session_state.data)
            st.session_state.numeric_columns = st.session_state.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        except Exception as e:
            st.write(f"Erreur lors du chargement des données : {e}")

    # Si les données ont été chargées
    if "data" in st.session_state:
        st.write("Aperçu des données :")
        st.write(st.session_state.data.head())
        
        st.write("Analyse descriptive des données :")
        afficher_analyse_descriptive(st.session_state.data)

        
        # Sélectionner la colonne pour afficher la distribution
        st.subheader("Distribution des colonnes")
        all_columns = st.session_state.data.columns.tolist()
        column_to_plot = st.selectbox("Choisissez une colonne à visualiser :", all_columns)
        if column_to_plot:
            afficher_distribution_numeriques(st.session_state.data, column_to_plot)

                # Matrice de corrélation
        if st.button("Afficher la matrice de corrélation"):
            afficher_matrice_correlation(st.session_state.data)
            
        # Traitement des valeurs manquantes
        st.subheader("Traitement des valeurs manquantes")
        missing_values = st.session_state.data.isnull().sum()
        columns_with_nan = missing_values[missing_values > 0].index.tolist()
        
        if columns_with_nan:
            for column in columns_with_nan:
                st.write(f"Colonne : {column}")
                treatment_option = st.selectbox(
                    f"Choisissez la méthode de traitement pour {column} :",
                    ["Moyenne", "Médiane", "Mode", "Supprimer la colonne", "Supprimer les lignes", "Validation croisée"],
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
        categorical_columns = st.selectbox("Choisissez une colonne catégorielle à encoder :", all_columns)
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
        
        if "results" in st.session_state:
            st.subheader("Résultats de Prédiction")
            st.write(st.session_state.results)
            
            csv = st.session_state.results.to_csv(index=False).encode('utf-8')
            st.download_button(label="Télécharger les résultats en CSV", data=csv, file_name='predictions.csv', mime='text/csv')
    else:
        st.write("Veuillez télécharger un fichier pour commencer.")
# Exécution de l'application
if __name__ == '__main__':
    main()
