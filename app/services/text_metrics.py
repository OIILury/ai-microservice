def calculer_distance_levenshtein(texte_a: str, texte_b: str) -> int:
    """
    Calcule la distance de Levenshtein (au niveau caractère) entre deux textes.
    Implémentation par programmation dynamique avec optimisation mémoire (2 lignes).
    """
    if len(texte_a) < len(texte_b):
        return calculer_distance_levenshtein(texte_b, texte_a)

    if len(texte_b) == 0:
        return len(texte_a)

    previous_row = list(range(len(texte_b) + 1))
    for i, c1 in enumerate(texte_a):
        current_row = [i + 1]
        for j, c2 in enumerate(texte_b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def compter_mots(texte: str) -> int:
    """Compte le nombre de mots dans un texte (split sur les espaces, filtre les chaînes vides)."""
    if not texte:
        return 0
    return len([w for w in texte.split() if w.strip()])

def calculer_ratio_taille(texte_entree: str, texte_sortie: str) -> float:
    """Retourne len(texte_sortie) / len(texte_entree), ou 0.0 si texte_entree est vide."""
    if not texte_entree:
        return 0.0
    return len(texte_sortie) / len(texte_entree)
