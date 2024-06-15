#!/usr/bin/env python
# coding: utf-8

# ## Importer les bibliothèques nécessaires

# In[91]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import requests
import streamlit as st


# ## Charger les données

# In[24]:


# Lire les données depuis un fichier CSV avec ';' comme séparateur
# URL du dataset
url = "https://opendata.paris.fr/explore/dataset/velib-disponibilite-en-temps-reel/download/?format=csv&timezone=Europe/Berlin&lang=fr"

# Télécharger le contenu du fichier CSV
response = requests.get(url)
data = response.content.decode('utf-8')

# Charger les données dans un DataFrame pandas
df = pd.read_csv(StringIO(data), delimiter=';')

# Afficher les premières lignes du DataFrame
print(df.head())


# In[40]:


df.drop(columns=['code_insee_commune'], inplace=True)
print (df)


# ## Exploration initiale des données

# #### Aperçu des données

# In[41]:


# Afficher les premières lignes du dataset
print(df.head())


# In[42]:


# Afficher les informations générales sur le dataset
print(df.info())


# In[43]:


# Statistiques descriptives
print(df.describe())


# In[44]:


df.describe().transpose()


# In[45]:


#Afficher le nb de lignes et colonnes
df.shape
# Afficher le nombre de lignes et de colonnes du DataFrame
print(f"Le DataFrame contient {df.shape[1]} colonnes et {df.shape[0]} lignes.")


# ## Exploration initiale des données

# #### Détection des valeurs manquantes

# In[46]:


# Vérifier les valeurs manquantes
missing_values = df.isnull().sum()
print("Valeurs manquantes par colonne :")
print(missing_values)


# In[47]:


# Pourcentage de valeurs manquantes par colonne
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)


# ## Gestion des valeurs manquantes

# In[48]:


df.head()


# In[49]:


df.isnull()


# ### Imputation par la valeur normalement attendue

# In[51]:


# Imputation par la valeur normalement attendue

# Pour 'Nom station', remplacer par une valeur constante ou une chaîne vide
df['name'].fillna('Non spécifié', inplace=True)

# Pour 'Capacité de la station', remplacer par la moyenne ou la médiane
df['capacity'].fillna(df['capacity'].median(), inplace=True)

print("DataFrame après imputation:")
print(df)


# In[52]:


# Vérifier les valeurs manquantes
missing_values = df.isnull().sum()
print("Valeurs manquantes par colonne :")
print(missing_values)


# ### Manipulation des filtres 

# In[54]:


def filter_by_column(df, column, condition):
  
    filtered_df = df[df[column].apply(lambda x: eval(f'x {condition}'))]
    return filtered_df


# In[56]:


filtered_df = filter_by_column(df, 'capacity', '>= 20')
print(filtered_df.head())


# ### GroupBy avec Comptage 

# In[60]:


# GroupBy par ville avec comptage des stations en fonctionnement
count_df = df[df['is_installed'] == 'OUI'].groupby('nom_arrondissement_communes').size().reset_index(name='Nbr de station en fct')

print("Nombre de stations en fonctionnement par ville:")
print(count_df)


# ### Création des deux nouvelles variables

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
df


# In[73]:


# Vérifier les valeurs manquantes
missing_values = df.isnull().sum()
print("Valeurs manquantes par colonne :")
print(missing_values)


# In[74]:


df.head()


# ## Visualisation des tendances 

# ### Histogrammes

# In[75]:


# Histogramme de la Capacité de la station
# Assurez-vous que les données sont dans un DataFrame appelé 'df'
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df['capacity'], bins=10, kde=True, ax=ax)
ax.set_title('Distribution de la Capacité de la station')
ax.set_xlabel('Capacité de la station')
ax.set_ylabel('Fréquence')
plt.show()

# Si vous utilisez Streamlit
import streamlit as st
st.pyplot(fig)


# ### Diagrammes circulaires (Pie charts)

# In[76]:


# Diagramme circulaire de la répartition des bornettes libres
plt.figure(figsize=(8, 8))
labels = ['Bornettes libres', 'Bornettes occupées']
sizes = [df['numdocksavailable'].sum(), df['capacity'].sum() - df['numdocksavailable'].sum()]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Répartition des bornettes libres')
plt.axis('equal')
plt.show()
st.pyplot(fig)


# ### Diagrammes de dispersion (Scatter plots)

# In[77]:


# Diagramme de dispersion entre Capacité de la station et Nombre total de vélos disponibles
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='capacity', y='numbikesavailable', hue='nom_arrondissement_communes', sizes=(20, 100), legend='auto')
plt.title('Relation entre Capacité de la station et Nombre total de vélos disponibles')
plt.xlabel('Capacité de la station')
plt.ylabel('Nombre total de vélos disponibles')
plt.show()
st.pyplot(fig)


# In[78]:


# Exemple : Relation entre Capacité de la station et Nombre total vélos disponibles
plt.figure(figsize=(10, 6))
sns.scatterplot(x='capacity', y='numbikesavailable', data=df, hue='is_installed', palette='Set2')
plt.title('Relation entre Capacité de la station et le nombre de station en fonctionnement')
plt.xlabel('Capacité de la station')
plt.ylabel('Nombre total vélos disponibles')
plt.legend(title='Station en fonctionnement')
plt.grid(True)
plt.show()
st.pyplot(fig)


# In[80]:


# Exemple pour la distribution de Station en fonctionnement
plt.figure(figsize=(8, 5))
sns.countplot(x='is_installed', data=df)
plt.title('Répartition des stations par fonctionnement')
plt.xlabel('Station en fonctionnement')
plt.ylabel('Nombre de stations')
plt.show()
st.pyplot(fig)


# In[81]:


# Sélection des colonnes pour la visualisation
stations = df['nom_arrondissement_communes']  # Assumant que 'Nom communes équipées' contient les noms des stations
vélos_disponibles = df['numbikesavailable']

# Création du graphique à barres
plt.figure(figsize=(15, 6))
plt.bar(stations, vélos_disponibles, color='skyblue')
plt.xlabel('Nom communes équipées')
plt.ylabel('Nombre total de vélos disponibles')
plt.title('Nombre de vélos disponibles par station')
plt.xticks(rotation=80, ha='right')  # Rotation des étiquettes de l'axe x pour une meilleure lisibilité
plt.tight_layout()

# Affichage du graphique
plt.show()
st.pyplot(fig)


# In[83]:


# Groupement des données par commune équipée
grouped = df.groupby('nom_arrondissement_communes').sum()

# Sélection des colonnes pertinentes
grouped = grouped[['mechanical', 'ebike']]

# Création du graphique
grouped.plot(kind='bar', figsize=(12, 8))
plt.title('Comparaison des vélos mécaniques et électriques disponibles par commune équipée')
plt.xlabel('Commune équipée')
plt.ylabel('Nombre de vélos')
plt.xticks(rotation=45)
plt.legend(['Vélos mécaniques disponibles', 'Vélos électriques disponibles'])
plt.tight_layout()
plt.show()
st.pyplot(fig)

