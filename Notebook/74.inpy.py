def supprimer_commentaires(chemin_fichier):
    with open(chemin_fichier, 'r', encoding='utf-8') as fichier:
        lignes = fichier.readlines()

    # Supprime les lignes qui commencent par #
    lignes_sans_commentaires = [ligne for ligne in lignes if not ligne.lstrip().startswith('#')]

    # Écrit les lignes modifiées dans le fichier
    with open(chemin_fichier, 'w', encoding='utf-8') as fichier:
        fichier.writelines(lignes_sans_commentaires)

# Exemple d'utilisation
supprimer_commentaires("final.py")

