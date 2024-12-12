import csv

def normalize_csv(input_file, output_file):
    """
    Corrige un fichier CSV en supprimant la colonne d'index si elle est présente.

    :param input_file: Chemin du fichier CSV d'entrée.
    :param output_file: Chemin du fichier CSV de sortie corrigé.
    """
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    # Vérifier et normaliser les colonnes
    normalized_rows = []
    for row in rows:
        if len(row) == 6:  # Ligne avec une colonne d'index supplémentaire
            normalized_rows.append(row[1:])  # Ignorer la première colonne
        elif len(row) == 5:  # Ligne correctement formatée
            normalized_rows.append(row)
        else:
            raise ValueError(f"Ligne avec un nombre inattendu de colonnes: {row}")

    # Écrire le fichier corrigé
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        # Écrire les en-têtes corrigés
        writer.writerow(["Image Name", "x1", "y1", "x2", "y2"])
        # Écrire les données corrigées
        writer.writerows(normalized_rows)

# Exemple d'utilisation
normalize_csv('wene.csv', 'fixed.csv')
