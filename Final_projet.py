#!/usr/bin/env python
# coding: utf-8

# ## Importation des bibliothèques nécesssaires

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import requests
import streamlit as st


# ## Configuration Streamlit et CSS

# In[ ]:


st.set_page_config(
    page_title="Projet Data Management", initial_sidebar_state="collapsed"
)
st.markdown("# Tim-Younes Jelinek, Mohamed-Amine AMMAR")


# In[2]:


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


# ## Imputation du jeu de données

# In[3]:


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


# ## Exploration initiales des données

# #### Aperçu des données

# In[4]:


df.shape
df.info()
df.describe().transpose()


# #### Détection des valeurs manquantes

# In[6]:


# Vérifier les valeurs manquantes
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

print("Valeurs manquantes par colonne : "),  print(missing_values)
print("Valeurs manquantes en pourcentages :"), print(missing_percentage)


# #### Gestion des valeurs manquantes

# In[ ]:


#On supprime les colonnes code insee commune et code geo qui ne contiennent que des valeurs vides et/ou ne contiennent pas d'éléments nécessaires au bon fonctionnement.
df.drop(columns=['code_insee_commune'], inplace=True)
df.drop(columns=['coordonnees_geo'], inplace=True)
df.head()


# ## Création de nouvelles variables

# #### Variable : fréquentation

# In[13]:


# Calculer le pourcentage de vélos disponibles par rapport à la capacité
df['% de vélos disponibles'] = (df['numbikesavailable'] / df['capacity']) * 100

# Définir la popularité en fonction du pourcentage de vélos disponibles
def classify_popularity(percentage):
    if percentage < 25:
        return 'Très populaire'
    elif percentage < 50:
        return 'Populaire'
    else:
        return 'Impopulaire'

df['Fréquentation'] = df['% de vélos disponibles'].apply(classify_popularity)

df['% de vélos disponibles'] = df['% de vélos disponibles'].fillna(0)


# #### Variable : taux de vélos électriques

# In[14]:


# Calculer le pourcentage de vélos électriques
df['% électrique'] = (df['ebike'] / df['numbikesavailable']) * 100
df['% électrique'] = df['ebike'].fillna(0)

# Afficher les premières lignes du DataFrame pour vérifier
df.head()

df['vélo hors d\'usage'] = df['capacity'] - (df['numdocksavailable'] + df['numbikesavailable'])
df['vélo hors d\'usage'] = df['vélo hors d\'usage'].fillna(0)
# Afficher les premières lignes du DataFrame pour vérifier
df.head()
# ## Affichage sur Streamlit

# #### Description et explications

# In[18]:


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


# #### Visualisation 

# ##### Graph 1 et 2

# In[ ]:


# Problème avec des valeurs négatives donc elles sont remplacées ou filtrées
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


# ##### Graph 3

# In[ ]:


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


# ##### Graph 4

# In[ ]:


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

# Affichage du graphique
st.pyplot(fig6)


# ##### Graph 5

# Initialiser session_state pour les villes et types de vélos si elles n'existent pas
if 'villes' not in st.session_state:
    st.session_state.villes = list(df['nom_arrondissement_communes'].unique())
if 'types_velos' not in st.session_state:
    st.session_state.types_velos = ['vélo hors d\'usage', 'numbikesavailable']

# Ajouter un titre au-dessus du formulaire
st.markdown('<div class="stTitle">Répartition en % du nombre de vélos hors d\'usage et disponibles par ville</div>', unsafe_allow_html=True)

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
        options=['vélo hors d\'usage', 'numbikesavailable'],
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
        if 'velo_hors_d_usage' in st.session_state.types_velos:
            hors_usage_percentage = (data_ville['vélo hors d\'usage'].sum() / total_capacity) * 100 if total_capacity > 0 else 0
            ax.barh(ville, hors_usage_percentage, color='skyblue', label='Vélos hors d\'usage' if ville == st.session_state.villes[0] else "")
            bottom += hors_usage_percentage
        if 'numbikesavailable' in st.session_state.types_velos:
            available_percentage = (data_ville['numbikesavailable'].sum() / total_capacity) * 100 if total_capacity > 0 else 0
            ax.barh(ville, available_percentage, left=bottom, color='lightgreen', label='Vélos disponibles' if ville == st.session_state.villes[0] else "")

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
# ## Corpus et exploration

# In[26]:


from unidecode import unidecode
import re
from nltk.stem import SnowballStemmer


# In[27]:


corpus = ["l'histogramme montre que la majorité des stations ont une capacité concentrée autour de 25-35",
"60.9% des bornettes sont libres, tandis que 39.1% des bornettes sont occupées",
"En 2024, on observe une augmentation notable du nombre de stations, marquant ainsi une expansion significative du réseau !!",
"La ville de Paris possède un nombre nettement supérieur de vélos disponibles par rapport aux autres villes",]
print(f"Les corpus est composé de {len(corpus)} documents")


# In[28]:


i = 1

for doc in corpus:
    print(f"Le document {i} est de longueur : {len(doc)}")
    i+=1


# In[29]:


i = 1

for doc in corpus:
    liste_mots = doc.split() # Permet de séparer les mots d'une chaines de caractère en fonction d'un séparateur 
                             # par défault l'espace.
        
    print(f"Le document {i} contient {len(liste_mots)} mots")
    print(f"La liste de mots du documents {i} : {liste_mots} \n")
    i+=1


# In[30]:


# Passage du corpus en minuscule
corpus_2 = [] # initialisation de la liste

for doc in corpus:
    corpus_2.append(doc.lower()) 
    
corpus_2


# In[31]:


# Assurer que chaque document est bien de type str
i = 1

for doc in corpus:
    if type(doc) == str:
        print(f"Le type du document {i} est {type(doc)}, c'est ok!" )
    else :
        print(f"Le type du document {i} est {type(doc)}, il faut le convertir!" )
        
    i+=1


# In[32]:


#suppression des accents
corpus = [unidecode(doc) for doc in corpus]

corpus


# In[34]:


# Définir une fonction pour effectuer les transformations
def preprocess_text(text):
    # Transformer les quatre chiffres en 'année'
    text = re.sub(r'\b\d{4}\b', 'année', text)

    # Transformer '60.9%' et '39.1%' par 'pourcentage'
    text = re.sub(r'60\.9%', 'pourcentage', text)
    text = re.sub(r'39\.1%', 'pourcentage', text)

    # Transformer '25-35' par 'chiffre'
    text = re.sub(r'25-35', 'chiffre', text)

    return text

# Appliquer la fonction preprocess_text à chaque élément du corpus
corpus_preprocessed = [preprocess_text(text) for text in corpus]

# Afficher le corpus après prétraitement
for text in corpus_preprocessed:
    print(text)


# In[36]:


#Suppression des numériques et caractères spéciaux
corpus = [re.sub(r"[^a-z]+", ' ', doc) for doc in corpus]

corpus


# In[37]:


# Définition de la liste de stop words considérés (celle de spacy)
stopWords = ['a', 'abord', 'absolument', 'afin', 'ah', 'ai', 'aie', 'ailleurs', 'ainsi', 'ait', 'allaient', 'allo', 'allons', 
             'allô', 'alors', 'anterieur', 'anterieure', 'anterieures', 'apres', 'après', 'as', 'assez', 'attendu', 'au', 
             'aucun', 'aucune', 'aujourd', "aujourd'hui", 'aupres', 'auquel', 'aura', 'auraient', 'aurait', 'auront', 'aussi', 
             'autre', 'autrefois', 'autrement', 'autres', 'autrui', 'aux', 'auxquelles', 'auxquels', 'avaient', 'avais', 'avait', 
             'avant', 'avec', 'avoir', 'avons', 'ayant', 'bah', 'bas', 'basee', 'bat', 'beau', 'beaucoup', 'bien', 'bigre', 'boum', 
             'bravo', 'brrr', "c'", 'car', 'ce', 'ceci', 'cela', 'celle', 'celle-ci', 'celle-là', 'celles', 'celles-ci', 'celles-là', 
             'celui', 'celui-ci', 'celui-là', 'cent', 'cependant', 'certain', 'certaine', 'certaines', 'certains', 'certes', 'ces', 
             'cet', 'cette', 'ceux', 'ceux-ci', 'ceux-là', 'chacun', 'chacune', 'chaque', 'cher', 'chers', 'chez', 'chiche', 'chut', 
             'chère', 'chères', 'ci', 'cinq', 'cinquantaine', 'cinquante', 'cinquantième', 'cinquième', 'clac', 'clic', 'combien', 
             'comme', 'comment', 'comparable', 'comparables', 'compris', 'concernant', 'contre', 'couic', 'crac', 'c’', "d'", 'da', 
             'dans', 'de', 'debout', 'dedans', 'dehors', 'deja', 'delà', 'depuis', 'dernier', 'derniere', 'derriere', 'derrière', 
             'des', 'desormais', 'desquelles', 'desquels', 'dessous', 'dessus', 'deux', 'deuxième', 'deuxièmement', 'devant', 'devers', 
             'devra', 'different', 'differentes', 'differents', 'différent', 'différente', 'différentes', 'différents', 'dire', 
             'directe', 'directement', 'dit', 'dite', 'dits', 'divers', 'diverse', 'diverses', 'dix', 'dix-huit', 'dix-neuf', 
             'dix-sept', 'dixième', 'doit', 'doivent', 'donc', 'dont', 'douze', 'douzième', 'dring', 'du', 'duquel', 'durant', 'dès', 
             'désormais', 'd’', 'effet', 'egale', 'egalement', 'egales', 'eh', 'elle', 'elle-même', 'elles', 'elles-mêmes', 'en', 
             'encore', 'enfin', 'entre', 'envers', 'environ', 'es', 'est', 'et', 'etaient', 'etais', 'etait', 'etant', 'etc', 'etre', 
             'eu', 'euh', 'eux', 'eux-mêmes', 'exactement', 'excepté', 'extenso', 'exterieur', 'fais', 'faisaient', 'faisant', 'fait', 
             'façon', 'feront', 'fi', 'flac', 'floc', 'font', 'gens', 'ha', 'hein', 'hem', 'hep', 'hi', 'ho', 'holà', 'hop', 'hormis', 
             'hors', 'hou', 'houp', 'hue', 'hui', 'huit', 'huitième', 'hum', 'hurrah', 'hé', 'hélas', 'i', 'il', 'ils', 'importe', 
             "j'", 'je', 'jusqu', 'jusque', 'juste', 'j’', "l'", 'la', 'laisser', 'laquelle', 'las', 'le', 'lequel', 'les', 
             'lesquelles', 'lesquels', 'leur', 'leurs', 'longtemps', 'lors', 'lorsque', 'lui', 'lui-meme', 'lui-même', 'là', 'lès', 'l’', 
             "m'", 'ma', 'maint', 'maintenant', 'mais', 'malgre', 'malgré', 'maximale', 'me', 'meme', 'memes', 'merci', 'mes', 'mien', 'mienne', 
             'miennes', 'miens', 'mille', 'mince', 'minimale', 'moi', 'moi-meme', 'moi-même', 'moindres', 'moins', 'mon', 
             'moyennant', 'même', 'mêmes', 'm’', "n'", 'na', 'naturel', 'naturelle', 'naturelles', 'ne', 'neanmoins', 'necessaire', 
             'necessairement', 'neuf', 'neuvième', 'ni', 'nombreuses', 'nombreux', 'non', 'nos', 'notamment', 'notre', 'nous', 'nous-mêmes', 
             'nouveau', 'nul', 'néanmoins', 'nôtre', 'nôtres', 'n’', 'o', 'oh', 'ohé', 'ollé', 'olé', 'on', 'ont', 'onze', 'onzième', 'ore', 
             'ou', 'ouf', 'ouias', 'oust', 'ouste', 'outre', 'ouvert', 'ouverte', 'ouverts', 'où', 'paf', 'pan', 'par', 'parce', 'parfois', 
             'parle', 'parlent', 'parler', 'parmi', 'parseme', 'partant', 'particulier', 'particulière', 'particulièrement', 'pas', 'passé', 
             'pendant', 'pense', 'permet', 'personne', 'peu', 'peut', 'peuvent', 'peux', 'pff', 'pfft', 'pfut', 'pif', 'pire', 'plein', 'plouf', 
             'plus', 'plusieurs', 'plutôt', 'possessif', 'possessifs', 'possible', 'possibles', 'pouah', 'pour', 'pourquoi', 'pourrais', 'pourrait', 
             'pouvait', 'prealable', 'precisement', 'premier', 'première', 'premièrement', 'pres', 'probable', 'probante', 'procedant', 'proche', 
             'près', 'psitt', 'pu', 'puis', 'puisque', 'pur', 'pure', "qu'", 'quand', 'quant', 'quant-à-soi', 'quanta', 'quarante', 'quatorze', 
             'quatre', 'quatre-vingt', 'quatrième', 'quatrièmement', 'que', 'quel', 'quelconque', 'quelle', 'quelles', "quelqu'un", 'quelque', 
             'quelques', 'quels', 'qui', 'quiconque', 'quinze', 'quoi', 'quoique', 'qu’', 'rare', 'rarement', 'rares', 'relative', 'relativement', 
             'remarquable', 'rend', 'rendre', 'restant', 'reste', 'restent', 'restrictif', 'retour', 'revoici', 'revoilà', 'rien', "s'", 'sa', 
             'sacrebleu', 'sait', 'sans', 'sapristi', 'sauf', 'se', 'sein', 'seize', 'selon', 'semblable', 'semblaient', 'semble', 'semblent', 
             'sent', 'sept', 'septième', 'sera', 'seraient', 'serait', 'seront', 'ses', 'seul', 'seule', 'seulement', 'si', 'sien', 'sienne', 
             'siennes', 'siens', 'sinon', 'six', 'sixième', 'soi', 'soi-même', 'soit', 'soixante', 'son', 'sont', 'sous', 'souvent', 'specifique', 
             'specifiques', 'speculatif', 'stop', 'strictement', 'subtiles', 'suffisant', 'suffisante', 'suffit', 'suis', 'suit', 'suivant', 
             'suivante', 'suivantes', 'suivants', 'suivre', 'superpose', 'sur', 'surtout', 's’', "t'", 'ta', 'tac', 'tant', 'tardive', 'te', 
             'tel', 'telle', 'tellement', 'telles', 'tels', 'tenant', 'tend', 'tenir', 'tente', 'tes', 'tic', 'tien', 'tienne', 'tiennes', 
             'tiens', 'toc', 'toi', 'toi-même', 'ton', 'touchant', 'toujours', 'tous', 'tout', 'toute', 'toutefois', 'toutes', 'treize', 'trente', 
             'tres', 'trois', 'troisième', 'troisièmement', 'trop', 'très', 'tsoin', 'tsouin', 'tu', 'té', 't’', 'un', 'une', 'unes', 
             'uniformement', 'unique', 'uniques', 'uns', 'va', 'vais', 'vas', 'vers', 'via', 'vif', 'vifs', 'vingt', 'vivat', 'vive', 'vives', 
             'vlan', 'voici', 'voilà', 'vont', 'vos', 'votre', 'vous', 'vous-mêmes', 'vu', 'vé', 'vôtre', 'vôtres', 'zut', 'à', 'â', 'ça', 'ès', 
             'étaient', 'étais', 'était', 'étant', 'été', 'être', 'ô', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 
             'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'qu']

stopWords = [unidecode(sw) for sw in stopWords]


# In[38]:


corpus = [' '.join([word for word in doc.split() if word not in stopWords]) for doc in corpus]

corpus


# In[39]:


stemmer = SnowballStemmer('french') # Vous initialisez le stemmer en français

corpus = [" ".join([stemmer.stem(word) for word in doc.split()]) for doc in corpus]

corpus


# ## La vectorisation

# In[40]:


#Import des librairies utiles au module
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from collections import defaultdict


# In[41]:


corpus= ['histogramm montr majorit station capacit concentre autour',
 'bornet libr tand bornet occupe',
 'observ augment notabl nombr station marqu expans signif reseau',
 'vill aris possed nombr net superieur velos disponibl rapport vill']
mots = ' '.join(corpus) # Concaténation des documents renvoi une chaine de caractères

liste_mots = re.findall(r"\w+", mots) # Création de la liste contenant tous les mots de notre corpus
print(f"Ci-dessous, la liste de mots présents dans le corpus : \n{liste_mots}\n")

vocab = set(liste_mots)
print(f"Le vocabulaire est de taille {len(vocab)}")
print(f"Ci-dessous, le vocabulaire (mots uniques) : \n{vocab}\n")


# In[42]:


# Représentation Bag of Words : CountVectorizer1
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(X.toarray())


# In[43]:


print(vectorizer.get_feature_names_out())


# In[44]:


#Les N-grammes
# Trouver les bigrammes dans les quatres documents
for i, doc in enumerate(corpus):
    print(f"Document {i}: {doc}")
    # Trouver les bigrammes dans le document actuel
    bigrams = list(nltk.bigrams(doc.split()))
    print(f"Liste des bigrammes présents dans le document {i}: {bigrams}")
    print()  # Pour ajouter une ligne vide entre chaque document


# In[45]:


#Représentation tf-idf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print(X.toarray())


# In[46]:


print(vectorizer.get_feature_names_out())


# In[47]:


vectorizer = TfidfVectorizer(ngram_range=(1,2)) 
X = vectorizer.fit_transform(corpus)

print(X.toarray())


# In[48]:


print(f"Pour chaque ligne qui correspond à un doc, je retrouve les {len(vectorizer.get_feature_names_out())} features suivants: \n{vectorizer.get_feature_names_out()}")


# In[49]:


vectorizer = TfidfVectorizer(min_df=2) # vous conservez que les mots présents au moins deux fois dans le corpus
X = vectorizer.fit_transform(corpus)

print(X.toarray()) 


# In[50]:


print(vectorizer.get_feature_names_out())


# In[51]:


# Initialisation du dictionnaire
freq = defaultdict(int)

# Compte l'ocurrence de chaque mot du corpus
for mot in liste_mots:
    freq[mot] += 1
    
print(freq)


# In[52]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud, STOPWORDS


# In[55]:


labels = ['Bornettes libres', 'Bornettes occupées']
texte = ' '.join(labels)
wordcloud = WordCloud(width=900, height=500, background_color='beige').generate(texte)

plt.figure(figsize=(15, 5))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[61]:


from wordcloud import WordCloud
from PIL import Image
from io import BytesIO
texte = ' '.join(labels)

#Récupération du mask
url="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Symbole_de_v%C3%A9lo.png/800px-Symbole_de_v%C3%A9lo.png"
response = requests.get(url)
mask = np.array(Image.open(BytesIO(response.content)))
wordcloud = WordCloud(width=400, height=400, background_color='white', mask=mask, contour_width=3, contour_color='steelblue').generate(texte)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

