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


# ## Visualisation des tendances 

# ### Histogrammes

# In[160]:


# Histogramme de la Capacité de la station
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df['capacity'], bins=10, kde=True, ax=ax)
ax.set_title('Distribution de la Capacité de la station')
ax.set_xlabel('Capacité de la station')
ax.set_ylabel('Fréquence')
plt.show()
st.pyplot(fig)


# ### Diagrammes circulaires (Pie charts)

# In[98]:


# Diagramme circulaire de la répartition des bornettes libres
fig2 = plt.figure(figsize=(8, 8))
labels = ['Bornettes libres', 'Bornettes occupées']
sizes = [df['numdocksavailable'].sum(), df['capacity'].sum() - df['numdocksavailable'].sum()]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Répartition des bornettes libres')
plt.axis('equal')
plt.show()
st.pyplot(fig2)


# ### Diagrammes de dispersion (Scatter plots)

# In[99]:


# Diagramme de dispersion entre Capacité de la station et Nombre total de vélos disponibles
fig3 = plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='capacity', y='numbikesavailable', hue='nom_arrondissement_communes', sizes=(20, 100), legend='auto')
plt.title('Relation entre Capacité de la station et Nombre total de vélos disponibles')
plt.xlabel('Capacité de la station')
plt.ylabel('Nombre total de vélos disponibles')
plt.show()
st.pyplot(fig3)


# In[100]:


# Exemple : Relation entre Capacité de la station et Nombre total vélos disponibles
fig3, ax = plt.subplots(figsize=(10, 6))

# Création du diagramme de dispersion
sns.scatterplot(x='capacity', y='numbikesavailable', data=df, hue='is_installed', palette='Set2', ax=ax)
ax.set_title('Relation entre Capacité de la station et le nombre de station en fonctionnement')
ax.set_xlabel('Capacité de la station')
ax.set_ylabel('Nombre total vélos disponibles')
ax.legend(title='Station en fonctionnement')
ax.grid(True)
plt.show()

# Affichage du graphique dans Streamlit
st.pyplot(fig)


# In[105]:


# Exemple pour la distribution de Station en fonctionnement
fig4, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x='is_installed', data=df, ax=ax)
ax.set_title('Répartition des stations par fonctionnement')
ax.set_xlabel('Station en fonctionnement')
ax.set_ylabel('Nombre de stations')
plt.show()
st.pyplot(fig4)


# ## MES GRAPHS

# In[111]:


# Sélection des colonnes pour la visualisation
stations = df['nom_arrondissement_communes']  # Assumant que 'Nom communes équipées' contient les noms des stations
vélos_disponibles = df['numbikesavailable']

fig5, ax1 = plt.subplots(figsize=(15, 10))
ax1.barh(stations, vélos_disponibles, color='skyblue')
ax1.set_xlabel('Nombre total de vélos disponibles')
ax1.set_ylabel('Nom des communes ')
ax1.set_title('Nombre de vélos disponibles par station')

# Assurez un layout compact
plt.tight_layout()

# Affichage du graphique
st.pyplot(fig5)

# Affichage du graphique
plt.show()

# Calculer le nombre de stations par commune
station_count = df['nom_arrondissement_communes'].value_counts().reset_index()
station_count.columns = ['nom_arrondissement_communes', 'station_count']

# Calculer la moyenne de vélos par station pour chaque commune
avg_bikes_per_station = df.groupby('nom_arrondissement_communes')['numbikesavailable'].mean().reset_index()
avg_bikes_per_station.columns = ['nom_arrondissement_communes', 'avg_bikes_per_station']

# Fusionner les deux DataFrames
merged_df = pd.merge(station_count, avg_bikes_per_station, on='nom_arrondissement_communes')

st.markdown('<div class="stTitle">Diagramme de dispersion entre la moyenne de vélos par station et le nombre de stations par commune</div>', unsafe_allow_html=True)

# Diagramme de dispersion entre la moyenne de vélos par station et le nombre de stations par commune
fig8, ax = plt.subplots(figsize=(12, 8))
scatter_plot = sns.scatterplot(data=merged_df, x='avg_bikes_per_station', y='station_count', hue='nom_arrondissement_communes', sizes=(20, 100), legend='auto', ax=ax)

# Positionnement de la légende en dehors de la figure
plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=8)
plt.title('Relation entre la moyenne de vélo par stations et le nombre total de stations par ville')
plt.xlabel('Moyenne de vélos par stations')
plt.ylabel('Nombre de stations par commune')
plt.show()
plt.show()
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
st.markdown('<div class="stTitle">Répartition des vélos mécaniques et électriques par ville</div>', unsafe_allow_html=True)

# Formulaire pour sélectionner les villes et les types de vélos
with st.form(key='form1'):
    st.write("Sélectionnez les paramètres pour le graphique 5")
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

    # Afficher les barres divisées pour chaque ville
    for ville in st.session_state.villes:
        data_ville = df[df['nom_arrondissement_communes'] == ville]
        bottom = 0
        if 'mechanical' in st.session_state.types_velos:
            ax.barh(ville, data_ville['mechanical'].sum(), color='skyblue', label='Vélos mécaniques' if ville == st.session_state.villes[0] else "")
            bottom += data_ville['mechanical'].sum()
        if 'ebike' in st.session_state.types_velos:
            ax.barh(ville, data_ville['ebike'].sum(), left=bottom, color='lightgreen', label='Vélos électriques' if ville == st.session_state.villes[0] else "")

    # Réglages des axes et des titres
    ax.set_xlabel('Nombre total de vélos disponibles')
    ax.set_ylabel('Nom des communes équipées')
    ax.set_title('Nombre de vélos disponibles par commune')
    ax.legend()

    plt.tight_layout()

    # Encadrer le graphique avec le même style que le formulaire
    st.markdown('<div class="stGraph">', unsafe_allow_html=True)
    st.pyplot(fig7)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.write("Veuillez sélectionner au moins une ville et un type de vélo.")







