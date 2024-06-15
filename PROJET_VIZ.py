#!/usr/bin/env python
# coding: utf-8

# ## Importer les bibliothèques nécessaires

# In[229]:


import pandas as pd
import numpy as np


# ## Charger les données

# In[230]:


# Lire les données depuis un fichier CSV avec ';' comme séparateur
df = pd.read_csv('velib-disponibilite-en-temps-reel (1).csv', sep=';')


# In[231]:


print (df)


# ## Exploration initiale des données

# #### Aperçu des données

# In[232]:


# Afficher les premières lignes du dataset
print(df.head())


# In[233]:


# Afficher les informations générales sur le dataset
print(df.info())


# In[234]:


# Statistiques descriptives
print(df.describe())


# In[235]:


df.describe().transpose()


# In[1]:


#Afficher le nb de lignes et colonnes
velib.shape


# In[237]:


# Obtenir le nombre de colonnes et ligne 
# le nombre de colonnes
num_columns = velib.shape[1]  # shape[1] donne le nombre de colonnes

# le nombre de lignes
num_lignes = velib.shape[0]  # shape[0] donne le nombre de lignes

# Afficher le nombre de lignes et de colonnes du DataFrame
print(f"Le DataFrame contient {num_columns} colonnes et {num_lignes} lignes.")


# ## Exploration initiale des données

# #### Détection des valeurs manquantes

# In[238]:


# Vérifier les valeurs manquantes
missing_values = df.isnull().sum()
print("Valeurs manquantes par colonne :")
print(missing_values)


# In[239]:


# Pourcentage de valeurs manquantes par colonne
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)


# ## Gestion des valeurs manquantes

# In[240]:


df.head()


# In[241]:


df.isnull()


# ### Imputation par la valeur normalement attendue

# In[242]:


# Imputation par la valeur normalement attendue
# Par exemple, remplacer les valeurs manquantes dans 'Identifiant station' par une valeur spécifique
df['Identifiant station'].fillna(99999, inplace=True)

# Pour 'Nom station', remplacer par une valeur constante ou une chaîne vide
df['Nom station'].fillna('Non spécifié', inplace=True)

# Pour 'Capacité de la station', remplacer par la moyenne ou la médiane
df['Capacité de la station'].fillna(df['Capacité de la station'].median(), inplace=True)

# Pour 'Nom communes équipées', remplacer par la valeur la plus fréquente (mode)
df['Nom communes équipées'].fillna(df['Nom communes équipées'].mode()[0], inplace=True)

print("DataFrame après imputation:")
print(df)


# In[243]:


# Vérifier les valeurs manquantes
missing_values = df.isnull().sum()
print("Valeurs manquantes par colonne :")
print(missing_values)


# ### Manipulation des filtres 

# In[244]:


def filter_by_column(df, column, condition):
  
    filtered_df = df[df[column].apply(lambda x: eval(f'x {condition}'))]
    return filtered_df


# In[245]:


filtered_df = filter_by_column(df, 'Capacité de la station', '>= 20')
print(filtered_df.head())


# ### GroupBy avec Comptage 

# In[246]:


# GroupBy par ville avec comptage des stations en fonctionnement
count_df = df[df['Station en fonctionnement'] == 'OUI'].groupby('Nom communes équipées').size().reset_index(name='Nombre de stations en fonctionnement')

print("Nombre de stations en fonctionnement par ville:")
print(count_df)


# ### Création des deux nouvelles variables

# In[247]:


# Création de la variable 'Taux d'occupation'
df['Taux d\'occupation'] = (df['Nombre total vélos disponibles'] / df['Capacité de la station']) * 100

# Création de la variable 'Proportion de vélos électriques'
df['Proportion de vélos électriques'] = (df['Vélos électriques disponibles'] / df['Nombre total vélos disponibles']) * 100


# Formater les colonnes en pourcentages avec deux décimales
df['Taux d\'occupation'] = df['Taux d\'occupation'].map('{:.2f}%'.format)
df['Proportion de vélos électriques'] = df['Proportion de vélos électriques'].map('{:.2f}%'.format)




# In[248]:


df


# In[249]:


# Vérifier les valeurs manquantes
missing_values = df.isnull().sum()
print("Valeurs manquantes par colonne :")
print(missing_values)


# In[250]:


df.head()


# ## Visualisation des tendances 

# ### Histogrammes

# In[251]:


import matplotlib.pyplot as plt
import seaborn as sns

# Histogramme de la Capacité de la station
plt.figure(figsize=(8, 5))
sns.histplot(df['Capacité de la station'], bins=10, kde=True)
plt.title('Distribution de la Capacité de la station')
plt.xlabel('Capacité de la station')
plt.ylabel('Fréquence')
plt.show()


# ### Diagrammes circulaires (Pie charts)

# In[252]:


# Diagramme circulaire de la répartition des bornettes libres
plt.figure(figsize=(8, 8))
labels = ['Bornettes libres', 'Bornettes occupées']
sizes = [df['Nombre bornettes libres'].sum(), df['Capacité de la station'].sum() - df['Nombre bornettes libres'].sum()]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Répartition des bornettes libres')
plt.axis('equal')
plt.show()


# ### Diagrammes de dispersion (Scatter plots)

# In[253]:


# Diagramme de dispersion entre Capacité de la station et Nombre total de vélos disponibles
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Capacité de la station', y='Nombre total vélos disponibles', hue='Nom communes équipées', sizes=(20, 100), legend='auto')
plt.title('Relation entre Capacité de la station et Nombre total de vélos disponibles')
plt.xlabel('Capacité de la station')
plt.ylabel('Nombre total de vélos disponibles')
plt.show()


# In[254]:


# Exemple : Relation entre Capacité de la station et Nombre total vélos disponibles
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Capacité de la station', y='Nombre total vélos disponibles', data=df, hue='Station en fonctionnement', palette='Set2')
plt.title('Relation entre Capacité de la station et Nombre total vélos disponibles')
plt.xlabel('Capacité de la station')
plt.ylabel('Nombre total vélos disponibles')
plt.legend(title='Station en fonctionnement')
plt.grid(True)
plt.show()


# In[255]:


# Exemple pour la distribution de Station en fonctionnement
plt.figure(figsize=(8, 5))
sns.countplot(x='Station en fonctionnement', data=df)
plt.title('Répartition des stations par fonctionnement')
plt.xlabel('Station en fonctionnement')
plt.ylabel('Nombre de stations')
plt.show()


# In[271]:


# Sélection des colonnes pour la visualisation
stations = df['Nom communes équipées']  # Assumant que 'Nom communes équipées' contient les noms des stations
vélos_disponibles = df['Nombre total vélos disponibles']

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


# In[272]:


# Groupement des données par commune équipée
grouped = df.groupby('Nom communes équipées').sum()

# Sélection des colonnes pertinentes
grouped = grouped[['Vélos mécaniques disponibles', 'Vélos électriques disponibles']]

# Création du graphique
grouped.plot(kind='bar', figsize=(12, 8))
plt.title('Comparaison des vélos mécaniques et électriques disponibles par commune équipée')
plt.xlabel('Commune équipée')
plt.ylabel('Nombre de vélos')
plt.xticks(rotation=45)
plt.legend(['Vélos mécaniques disponibles', 'Vélos électriques disponibles'])
plt.tight_layout()
plt.show()


# In[280]:


# Exemple de création d'un dataframe avec vos données
data = {
    'Identifiant station': [12109, 16107, 14111, 32304, 14014],
    'Nom station': ['Mairie du 12ème', 'Benjamin Godard - Victor Hugo', 'Cassini - Denfert-Rochereau', 'Charcot - Benfleet', 'Jourdan - Stade Charléty'],
    'Vélos mécaniques disponibles': [22, 0, 0, 0, 1],
    'Vélos électriques disponibles': [4, 1, 3, 0, 9]
}

df = pd.DataFrame(data)

# Sélection des données à visualiser
stations = df['Nom station']
velos_mecaniques = df['Vélos mécaniques disponibles']
velos_electriques = df['Vélos électriques disponibles']

# Création du graphique
fig, ax = plt.subplots(figsize=(10, 6))

# Largeur des barres
bar_width = 0.35

# Position des barres
bar_positions = range(len(stations))

# Création des barres pour vélos mécaniques
bars1 = ax.bar(bar_positions, velos_mecaniques, bar_width, label='Vélos mécaniques')

# Création des barres pour vélos électriques
bars2 = ax.bar([p + bar_width for p in bar_positions], velos_electriques, bar_width, label='Vélos électriques')

# Réglages des axes et des titres
ax.set_xlabel('Stations')
ax.set_ylabel('Nombre de vélos disponibles')
ax.set_title('Comparaison des vélos mécaniques et électriques disponibles par station')
ax.set_xticks([p + bar_width / 2 for p in bar_positions])
ax.set_xticklabels(stations, rotation=45, ha='right')
ax.legend()

# Affichage du graphique
plt.tight_layout()
plt.show()


# In[ ]:




