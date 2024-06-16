#!/usr/bin/env python
# coding: utf-8

# ## Importer les bibliothèques nécessaires

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import requests
import streamlit as st


# ## Configuration Streamlit et CSS

# In[7]:


st.set_page_config(
    page_title="Projet Data Management", page_icon="🖼️", initial_sidebar_state="collapsed"
)
st.markdown("# Tim-Younes Jelinek, Mohamed-Amine AMMAR")


# In[ ]:


# CSS
page_bg_img = '''
<style>

.stButton > button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 5px;
    padding: 10px 20px;
}
.stForm {
    background-color: #363636;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.stTitle {
    font-size: 24px;
    font-weight: bold;
    color: black;
    margin-bottom: 10px;
}
</style>
'''
# Appliquer le CSS
st.markdown(page_bg_img, unsafe_allow_html=True)


# ## Charger les données

# In[9]:


# Lire les données depuis un fichier CSV avec ';' comme séparateur
# URL du dataset
url = "https://opendata.paris.fr/explore/dataset/velib-disponibilite-en-temps-reel/download/?format=csv&timezone=Europe/Berlin&lang=fr"

# Télécharger le contenu du fichier CSV
response = requests.get(url)
data = response.content.decode('utf-8')

# Charger les données dans un DataFrame pandas
df = pd.read_csv(StringIO(data), delimiter=';')

# Afficher les premières lignes du DataFrame
df.head()


# ## Exploration initiale des données

# #### Aperçu des données

# In[10]:


#Afficher le nb de lignes et colonnes
df.shape


# In[11]:


# Afficher les informations générales sur le dataset
df.info()


# In[155]:


df.describe().transpose()


# #### Détection des valeurs manquantes

# In[12]:


# Vérifier les valeurs manquantes
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

print("Valeurs manquantes par colonne : "),  print(missing_values)
print("Valeurs manquantes en pourcentages :"), print(missing_percentage)


# ## Gestion des valeurs manquantes

# In[13]:


#On supprime la colonne code insee commune qui ne contient que des valeurs vides et ne contient pas d'éléments nécessaires au bon fonctionnement.
df.drop(columns=['code_insee_commune'], inplace=True)
df.head()




# ## Création des deux nouvelles variables

# In[72]:


# Création de la variable 'Taux d'occupation'
df['Taux d\'occupation'] = (df['numbikesavailable'] / df['capacity']) * 100

# Création de la variable 'Proportion de vélos électriques'
def calculate_proportion(row):
    if row['is_installed'] == 'OUI':
        if row['numbikesavailable'] != 0:
            return (row['ebike'] / row['numbikesavailable']) * 100
        else:
            return 0
    else:
        return '0.00%'

df['Proportion de vélos électriques'] = df.apply(calculate_proportion, axis=1)

# Formater les colonnes en pourcentages avec deux décimales
df['Taux d\'occupation'] = df['Taux d\'occupation'].map('{:.2f}%'.format)
df['Proportion de vélos électriques'] = df['Proportion de vélos électriques'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)
df.head()


#################################################
st.markdown("## Description du jeu de données")
st.write(f"""Nous avons choisi ce jeu de données en raison de l'approche des Jeux Olympiques. Avec des métros saturés et de nouvelles lignes non encore livrées, nous avons décidé de nous concentrer sur un moyen de transport alternatif emblématique à Paris : le Vélib.
Avec **+ 10** vélos utilisés par seconde en heure de pointe, **100 000** utilisateurs par jour et utilisé dans une **soixantaine** de communes en Île-de-France, il était intéressant pour nous de voir si le réseau de vélibs pourrait soulager les transports franciliens
Notre étude portera sur [un jeu de données de la Mairie de Paris](https://opendata.paris.fr/explore/dataset/velib-disponibilite-en-temps-reel/information/?disjunctive.name&disjunctive.is_installed&disjunctive.is_renting&disjunctive.is_returning&disjunctive.nom_arrondissement_communes), qui traite de l'infrastructure du système Vélib.
Le jeu de données contient {df.shape[0]} observations""")
st.markdown("### Les types de variables")
st.write("""Nous allons d'abord expliquer les variables non-intuitives :""")
st.write("""**stationcode** correspond au code attribué à la station""")
st.write("""**name** correspond au nom de la station""")
st.write("""**is_installed** indique si la station est fonctionnelle ou non""")
st.write("""**capacity** indique le nombre de vélos que cette station peut acceuillir""")
st.write("""**numdocksavailable** indique le nombre de bornes libres""")
st.write("""**numbikeavalaiable** correspond au nombre de vélos disponibles""")
st.write("""**mechanical et ebike**, respectivement le nombre de vélos mécaniques et électriques""")
st.write("""**is_renting** indique si il ya une borne de paiement et **is_returning** indique si il est possible de retourner son vélib à cette station""")
st.write("""**duedate** correspond à la date de la dernière actualisation des informations de la station""")
st.write("""**fréquentation** correspond à la popularité de la station et **% de vélos électriques** est éponyme""")
st.dataframe(df)
st.markdown("## Statistiques descriptives")
st.dataframe(df.describe().transpose())

st.markdown("## Visualisation")

# ## MES GRAPHS

# Assurez-vous que les valeurs négatives sont remplacées ou filtrées
df['capacity'] = df['capacity'].clip(lower=0)
df['numbikesavailable'] = df['numbikesavailable'].clip(lower=0)

# Calculer la capacité totale des stations par commune
capacity_per_commune = df.groupby('nom_arrondissement_communes')['capacity'].sum()

# Calculer le nombre total de vélos disponibles par commune
bikes_available_per_commune = df.groupby('nom_arrondissement_communes')['numbikesavailable'].sum()

# Fusionner les deux séries en un DataFrame
comparison_data = pd.DataFrame({
    'Capacité totale': capacity_per_commune,
    'Vélos disponibles': bikes_available_per_commune
})

# Ajouter un titre au-dessus du graphique
st.markdown('<div class="stTitle">Analyse du nombre de vélibs en circulation</div>', unsafe_allow_html=True)

# Filtrer les données pour les communes sélectionnées (ou toutes les communes)
communes = comparison_data.index.tolist()
comparison_data_filtered = comparison_data.loc[communes]

# Création du graphique en barres horizontales groupées
fig9, ax = plt.subplots(figsize=(15, 10))

# Définir la largeur des barres
bar_width = 0.35

# Positions des barres
index = np.arange(len(comparison_data_filtered))

# Barres pour la capacité totale
bar1 = ax.barh(index, comparison_data_filtered['Capacité totale'], bar_width, label='Capacité totale', color='skyblue')

# Barres pour les vélos disponibles
bar2 = ax.barh(index + bar_width, comparison_data_filtered['Vélos disponibles'], bar_width, label='Vélos disponibles', color='lightgreen')

# Ajouter des étiquettes et un titre
ax.set_ylabel('Commune')
ax.set_xlabel('Nombre')
ax.set_title('Comparaison des capacités des stations et des vélos disponibles par commune')
ax.set_yticks(index + bar_width / 2)
ax.set_yticklabels(comparison_data_filtered.index, rotation=0)
ax.legend()

# Afficher le graphique dans Streamlit
st.pyplot(fig9)

# Calculer les totaux globaux
total_capacity = df['capacity'].sum()
total_bikes_available = df['numbikesavailable'].sum()
total_bikes_in_circulation = total_capacity - total_bikes_available

# Création du pie chart pour le pourcentage de vélos en circulation
fig10, ax2 = plt.subplots()

labels = ['Vélos en circulation', 'Vélos disponibles']
sizes = [total_bikes_in_circulation, total_bikes_available]
colors = ['#ff9999','#66b3ff']
explode = (0.1, 0)  # Explode the first slice

ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax2.set_title('Répartition des vélos en circulation et disponibles')

# Afficher le pie chart dans Streamlit
st.pyplot(fig10)


st.markdown('<div class="stTitle">Relation entre le nombre de stations fonctionnelles par commune et la moyenne de vélos par station</div>', unsafe_allow_html=True)

# Calculer le nombre de stations par commune
station_count = df['nom_arrondissement_communes'].value_counts().reset_index()
station_count.columns = ['nom_arrondissement_communes', 'station_count']

# Calculer la moyenne de vélos par station pour chaque commune
avg_bikes_per_station = df.groupby('nom_arrondissement_communes')['numbikesavailable'].mean().reset_index()
avg_bikes_per_station.columns = ['nom_arrondissement_communes', 'avg_bikes_per_station']

# Fusionner les deux DataFrames
merged_df = pd.merge(station_count, avg_bikes_per_station, on='nom_arrondissement_communes')

# Diagramme de dispersion entre la moyenne de vélos par station et le nombre de stations par commune
fig8, ax = plt.subplots(figsize=(12, 8))
scatter_plot = sns.scatterplot(data=merged_df, x='avg_bikes_per_station', y='station_count', hue='nom_arrondissement_communes', sizes=(20, 100), legend='auto', ax=ax)

# Positionnement de la légende en dehors de la figure
ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=8)
ax.set_title('Relation entre la moyenne de vélos par station et le nombre de stations par commune')
ax.set_xlabel('Moyenne de vélos par station')
ax.set_ylabel('Nombre de stations par commune')

st.pyplot(fig8)


df['duedate'] = df['duedate'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0]
df['duedate'] = pd.to_datetime(df['duedate'], format='%Y-%m-%d')

# Filtrer les stations qui ne sont pas en service
df_not_installed = df[df['is_installed'] == "NON"].copy()

# Extraire les mois et années des dates des dernières actualisations
df_not_installed['year_month'] = df_not_installed['duedate'].dt.to_period('M')

st.markdown('<div class="stTitle">Evolution nominale du nombre de stations fermées</div>', unsafe_allow_html=True)

# Filtre pour sélectionner la commune ou toutes les communes
communes = ['Toutes les communes'] + df_not_installed['nom_arrondissement_communes'].unique().tolist()
selected_commune = st.selectbox('Sélectionnez la commune', options=communes)

# Filtrer les données si une commune est sélectionnée
if selected_commune != 'Toutes les communes':
    df_not_installed = df_not_installed[df_not_installed['nom_arrondissement_communes'] == selected_commune]

# Compter le nombre de stations non en service par mois et par année
updates_per_month = df_not_installed['year_month'].value_counts().sort_index()

# Convertir les périodes en timestamps pour le formatage littéraire
updates_per_month.index = updates_per_month.index.to_timestamp()

# Création du graphique temporel
fig6, ax = plt.subplots(figsize=(15, 6))
ax.plot(updates_per_month.index, updates_per_month.values, marker='o', linestyle='-')
ax.set_xlabel('Date')
ax.set_ylabel('Nombre de stations plus en fonctionnement')
ax.set_title(f'Nombre de stations plus en fonctionnement par mois pour {selected_commune}')

# Formater les étiquettes de l'axe des x en mois et années littéraires
ax.set_xticks(updates_per_month.index)
ax.set_xticklabels(updates_per_month.index.strftime('%B %Y'), rotation=45, ha='right')
plt.show()

# Affichage du graphique
st.pyplot(fig6)


# Initialiser session_state pour les villes et types de vélos si elles n'existent pas
if 'villes' not in st.session_state:
    st.session_state.villes = list(df['nom_arrondissement_communes'].unique())
if 'types_velos' not in st.session_state:
    st.session_state.types_velos = ['mechanical', 'ebike']

# Ajouter un titre au-dessus du formulaire
st.markdown('<div class="stTitle">Répartition du nombre de vélos mécaniques et électriques par ville</div>', unsafe_allow_html=True)

# Formulaire pour sélectionner les villes et les types de vélos
with st.form(key='form1'):
    st.write("Sélectionnez les paramètres pour le graphique")
    villes = st.multiselect(
        "Sélectionnez les villes",
        options=list(df['nom_arrondissement_communes'].unique()),
        default=st.session_state.villes
    )

    types_velos = st.multiselect(
        "Sélectionnez le type de vélos",
        options=['mechanical', 'ebike'],
        default=st.session_state.types_velos
    )

    submit_button = st.form_submit_button(label='Envoyer')

if submit_button:
    # Mettre à jour les sélections dans session_state
    st.session_state.villes = villes
    st.session_state.types_velos = types_velos

# Créer le graphique pour les vélos disponibles par commune
if st.session_state.villes and st.session_state.types_velos:
    fig7, ax = plt.subplots(figsize=(15, 10))

    # Calculer les pourcentages pour chaque ville
    for ville in st.session_state.villes:
        data_ville = df[df['nom_arrondissement_communes'] == ville]
        total_capacity = data_ville['capacity'].sum()
        bottom = 0
        if 'mechanical' in st.session_state.types_velos:
            mechanical_percentage = (data_ville['mechanical'].sum() / total_capacity) * 100 if total_capacity > 0 else 0
            ax.barh(ville, mechanical_percentage, color='skyblue', label='Vélos mécaniques' if ville == st.session_state.villes[0] else "")
            bottom += mechanical_percentage
        if 'ebike' in st.session_state.types_velos:
            ebike_percentage = (data_ville['ebike'].sum() / total_capacity) * 100 if total_capacity > 0 else 0
            ax.barh(ville, ebike_percentage, left=bottom, color='lightgreen', label='Vélos électriques' if ville == st.session_state.villes[0] else "")

    # Réglages des axes et des titres
    ax.set_xlabel('Pourcentage du nombre total de vélos disponibles')
    ax.set_ylabel('Nom des communes équipées')
    ax.set_title('Pourcentage de vélos disponibles par commune')
    ax.legend()

    plt.tight_layout()

    # Encadrer le graphique avec le même style que le formulaire
    st.markdown('<div class="stGraph">', unsafe_allow_html=True)
    st.pyplot(fig7)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.write("Veuillez sélectionner au moins une ville et un type de vélo.")



