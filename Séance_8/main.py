#coding:utf8

import numpy as np
import pandas as pd
import scipy
import scipy.stats

def ouvrirUnFichier(nom):
    with open(nom, "r") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

def tableauDeContingence(nom, donnees):
    indexValeurs = {}
    for element in range(0,len(nom)):
        indexValeurs.update({element: nom[element]})
    return pd.DataFrame(donnees).rename(index = indexValeurs)

def sommeDesColonnes(tableau):
    colonne = list(tableau.head(0))
    sommeColonne = []
    for element in colonne:
        sommeColonne.append(tableau[element].sum())
    return sommeColonne

def sommeDesLignes(tableau):
    colonne = list(tableau.head(0))
    sommeLigne = []
    for element1 in range(0,len(tableau)):
        ligne = []
        for element2 in range(0,len(colonne)):
            ligne.append(tableau.iloc[element1, element2])
        sommeLigne.append(np.sum(list(ligne)))
    return sommeLigne

data = pd.DataFrame(ouvrirUnFichier("./data/Socioprofessionnelle-vs-sexe.csv"))
#print(data)

#Création du tableau de contingence
#Contrairement à l'usage, vous ne devez pas créer de tableau croisé dynamique, puisque le fichier est déjà un tableau de contingence
print("Colonnes du DataFrame :", data.columns.tolist())
tableauDeContingence = tableauDeContingence(data["CatÃ©gorie"], {"Femmes": data["Femmes"], "Hommes": data["Hommes"]})
print(tableauDeContingence)

#Calculer les marges
print("Calcul des marges")

# Marges des colonnes et des lignes
marges_colonnes = sommeDesColonnes(tableauDeContingence)
marges_lignes = sommeDesLignes(tableauDeContingence)

# Construire un nouveau tableau avec une colonne 'Total' pour les marges lignes
tableau_avec_marges = tableauDeContingence.copy()
tableau_avec_marges["Total"] = marges_lignes

# Ajouter une ligne 'Total' contenant les marges colonnes et le total général
grand_total = sum(marges_colonnes)
ligne_totaux = marges_colonnes + [grand_total]
tableau_avec_marges.loc["Total"] = ligne_totaux

# Vérification : le total des marges des lignes doit être égal au total des marges des colonnes
total_lignes = sum(marges_lignes)
total_colonnes = sum(marges_colonnes)
if total_lignes == total_colonnes:
    print(f"Les totaux des marges sont identiques : {total_lignes}")
else:
    print(f"Attention : totaux différents - marges lignes = {total_lignes}, marges colonnes = {total_colonnes}")

print(tableau_avec_marges)

#Faire le test du chi2 avec les outils Scipy.stats
print("Test du chi2")

# Test du chi2 d'indépendance
observed = tableauDeContingence.values
chi2_val, p_value, dof, expected = scipy.stats.chi2_contingency(observed)
print(f"chi2 = {chi2_val}, p-value = {p_value}, ddl = {dof}")

# Interprétation (seuil 0.05)
alpha = 0.05
if p_value < alpha:
    print(f"Rejet de l'indépendance (p < {alpha}) → Il existe une liaison.")
else:
    print(f"On ne rejette pas l'indépendance (p >= {alpha}) → Pas de liaison détectée.")

#Calculer l'intensité de liaison phi2 de Pearson et Cramér's V
n = observed.sum()
phi2 = chi2_val / n
rows, cols = observed.shape
cramers_v = np.sqrt(chi2_val / (n * (min(rows, cols) - 1)))
print(f"phi2 = {phi2}")
print(f"Cramér's V = {cramers_v}")

