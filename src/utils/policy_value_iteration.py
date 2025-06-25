""" Première mission de la semaine 5 mai au 12 mai
Objectifs : - Écrire le code pour trouver la valeur d'une politique dans un environnement totalement connu
            - Développer un Algorithme d'apprentissage sans avoir recours à la matrice de transition ni aux récompenses
            - Programmer le jeu de 3x4 cases pour mieux voir la simulation."""
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.preprocessing import normalize

def valeur_politique(pi,R,P,gamma):
    """ pi : la politique, une liste d'action en forme d'indice ( on utilise les keys )
        R : matrice de récompense où chaque colonne j correspond à un vecteur
        de recompense associé à l'action pi(j) numpy de dimension nbetat x nbaction
        P : matrice de transitions controlée à 3 dimensions en utilisant le package
         numpy de dimension nbetat x nbaction x nbetat
        gamma : un réel de ]0,1[ pour avoir la convergence de l'algorithme """
    # Idée : créer une matrice Ppi dont chaque ligne est la ligne P(i,pi(i),.) et
    # un vecteur de récompense Rpi dont chaque coeff correspond R(i,pi(i))
    # et après resoudre (I-gamma*Ppi)^-1 * Rpi

    nb_etat = P.shape[0]
    if len(pi) != nb_etat:
        raise ValueError(f"La longueur de la politique pi ({len(pi)}) doit être égale au nombre d'états ({nb_etat}).")
    nb_action = P.shape[1]
    if np.shape(R) != (nb_etat,nb_action):
        raise ValueError(f"La matrice de récompense R doit être ({nb_etat},{nb_action})")
    if not(0 < gamma <1) :
        raise ValueError(f"La valeur gamma = {gamma} n'est pas dans ]0,1[")
    P_pi = np.zeros((nb_etat,nb_etat))
    R_pi = np.zeros(nb_etat)
    I = np.eye(nb_etat)
    for i in range(nb_etat):
        P_pi[i,:] = P[i,pi[i],:]
        R_pi[i] = R[i,pi[i]]

    V_pi = np.linalg.solve((I-gamma*P_pi),R_pi)
    return V_pi

def valeur_politique_livre(pi_non_terminal, R, P, gamma,
                           indices_etats_non_terminaux,
                           idx_etat_plus_1, val_etat_plus_1,
                           idx_etat_moins_1, val_etat_moins_1):
    """ pi_non_terminal : Politique pour les états NON TERMINAUX uniquement.
        R : Matrice R(s,a) pour TOUS les états (11x4).
        P: Matrice de transition pour tous les états (11x4x11).
        gamma : Facteur d'actualisation.
        indices_etats_non_terminaux : Liste des indices (0-10) des états non terminaux.
        idx_etat_plus_1 : Index global (0-10) de l'état terminal +1.
        val_etat_plus_1 : Valeur de l'état terminal +1 .
        idx_etat_moins_1 : Index global (0-10) de l'état terminal -1.
        val_etat_moins_1 : Valeur de l'état terminal -1."""
    nb_etats_total = P.shape[0]
    nb_etats_non_terminaux = len(indices_etats_non_terminaux)

    if len(pi_non_terminal) != nb_etats_non_terminaux:
        raise ValueError(f"La politique pi ({len(pi_non_terminal)}) doit être de taille {nb_etats_non_terminaux} (nb d'états non terminaux).")

    # P_pi_non_term ne contiendra que les transitions entre états non terminaux
    P_pi_non_term = np.zeros((nb_etats_non_terminaux, nb_etats_non_terminaux))
    # R_pi_non_term contiendra les récompenses modifiées pour les états non terminaux
    R_pi_non_term = np.zeros(nb_etats_non_terminaux)

    for i, idx in enumerate(indices_etats_non_terminaux):

        action_idx = pi_non_terminal[i]
        recompense = R[idx, action_idx]
        R_pi_non_term[i] = recompense
        if P[idx, action_idx, idx_etat_plus_1] > 0:
            R_pi_non_term[i] += gamma * P[idx, action_idx, idx_etat_plus_1] * val_etat_plus_1
        if P[idx, action_idx, idx_etat_moins_1] > 0:
            R_pi_non_term[i] += gamma * P[idx, action_idx, idx_etat_moins_1] * val_etat_moins_1

        for j, s in enumerate(indices_etats_non_terminaux):
            P_pi_non_term[i, j] = P[idx, action_idx, s]

    I = np.eye(nb_etats_non_terminaux)
    V_non_terminaux = np.linalg.solve((I - gamma * P_pi_non_term), R_pi_non_term)

    # Reconstruire le vecteur V complet
    V_complet = np.zeros(nb_etats_total)
    V_complet[idx_etat_plus_1] = val_etat_plus_1
    V_complet[idx_etat_moins_1] = val_etat_moins_1
    for i, idx in enumerate(indices_etats_non_terminaux):
        V_complet[idx] = V_non_terminaux[i]

    return V_complet

# Application sur Grid World

# Définition des cases
coordonnees = [] # les coordonnées de chaque cases si accessible

for i in range(3): # l'ordonné en commencant par 0 pour la première case en haut à gauche
    for j in range(4): # l'abscisse
        if not (i==1 and j == 1): # on exclut la case (1,1)
            coordonnees.append((i,j))

# Mappage  états -> index et index -> états
etat_en_index = {j:i for i,j in enumerate(coordonnees)} # pour mieux parcourir la matrice de transition
index_en_etat = {i:j for i,j in enumerate(coordonnees)} # pour se repérer dans la grille
nb_etat = len(coordonnees)  # nombre d'états : 11

idx_etat_plus_1 = etat_en_index[(0,3)]   # Devrait être 3
idx_etat_moins_1 = etat_en_index[(1,3)]  # Devrait être 6
val_etat_plus_1 = 1.0
val_etat_moins_1 = -1.0

indices_etats_terminaux = [idx_etat_plus_1, idx_etat_moins_1]
indices_etats_non_terminaux = [i for i in range(nb_etat) if i not in indices_etats_terminaux]
# Ceci donnera [0, 1, 2, 4, 5, 7, 8, 9, 10] (9 états non terminaux)
nb_etats_non_terminaux = len(indices_etats_non_terminaux)

# Action
action = {
    0:(-1,0), # en haut
    1:(1,0),  # en bas
    2:(0,-1), # à gauche
    3:(0,1)   # à droite
}

nb_action = len(action)

# Récompense
R = np.full((nb_etat, nb_action), -0.04)
for s_term_idx in indices_etats_terminaux:
    R[s_term_idx, :] = 0.0

action_perp = {  # les actions perpendiculaires à une action
    0 :(2,3), # en haut -> action perpendiculaire à gauche = gauche, action perpendiculaire à droite = droite
    1 :(3,2), # en bas -> action perpendiculaire à gauche = droite, action perpendiculaire à droite = gauche
    2 :(1,0), # à gauche -> action perpendiculaire à gauche = bas, action perpendiculaire à droite = haut
    3 :(0,1)  # à droite -> action perpendiculaire à gauche = haut, action perpendiculaire à droite = bas
}

# Matrice de transition
P = np.zeros((nb_etat,nb_action,nb_etat)) # matrice tri-dimensionnelle de taille nbetat x nbaction x nbetat initialisé à 0

# Obstacle et dimensions de la grille
obstacle_coord = (1, 1)
lignes_grille, cols_grille = 3, 4

for etat_courant_idx in range(nb_etat):
    etaty_courant, etatx_courant = index_en_etat[etat_courant_idx]

    for action_courant_idx in range(nb_action):

        sorties = [
            (action_courant_idx, 0.8),  # Mouvement intentionnel
            (action_perp[action_courant_idx][0], 0.1),  # Perpendiculaire gauche
            (action_perp[action_courant_idx][1], 0.1)   # Perpendiculaire droit
        ]
        for action_idx, prob in sorties:
            # Pour chaque sortie, on prend l'action
            actiony, actionx = action[action_idx]
            # et on regarde la case tenté par cette action
            etaty_tente, etatx_tente = etaty_courant + actiony, etatx_courant + actionx

            # On détermine la case résultante
            etaty_resultant, etatx_resultant = etaty_courant, etatx_courant  # État résultant par défaut: rester sur place (collision)
            etat_resultant_coord = (etaty_resultant, etatx_resultant) # Coordonnée de l'état résultant

            # On vérifie si la case tentée est valide
            if (0 <= etaty_tente < lignes_grille and
                0 <= etatx_tente < cols_grille and
                (etaty_tente, etatx_tente) != obstacle_coord):
                # Mouvement valide, la case résultante est la case tentée
                etaty_resultant, etatx_resultant = etaty_tente, etatx_tente
                etat_resultant_coord = (etaty_resultant, etatx_resultant)

            etat_resultant_idx = etat_en_index[etat_resultant_coord]
            P[etat_courant_idx, action_courant_idx, etat_resultant_idx] += prob

for s_term_idx in indices_etats_terminaux:
    for a_idx in range(nb_action):
        P[s_term_idx, a_idx, :] = 0.0 # Aucune transition vers d'autres états
        P[s_term_idx, a_idx, s_term_idx] = 1.0 # Boucle sur lui-même

#print(coordonnees)
#print(P[:,0,:])

# Vérification
toutbon=0
for s_idx in range(nb_etat):
    for a_idx in range(nb_action):
        sum_probs = np.sum(P[s_idx, a_idx, :])
        if not np.isclose(sum_probs, 1.0):
            coord_s = index_en_etat[s_idx]
            toutbon = 1
            print(f"Erreur: Somme des probabilités != 1 pour état {coord_s} (idx {s_idx}), action {a_idx}. Somme = {sum_probs:.2f}")
#if toutbon == 0:
#    print("TOUT EST BON POUR LA MATRICE")

# test de valeur de la politique: toujours en haut
#pi=[0]*9
# politique optimale du chapitre 17
pi_opti = [
     3, 3, 3,  # (0,0)->D, (0,1)->D, (0,2)->D  (vers +1)
     0, 2,     # (1,0)->H, (1,2)->G
     0, 2, 2, 0 # (2,0)->H, (2,1)->G, (2,2)->G, (2,3)->H
]
gamma = 1
V = valeur_politique_livre(pi_opti,R,P,gamma,indices_etats_non_terminaux,
    idx_etat_plus_1, val_etat_plus_1,
    idx_etat_moins_1, val_etat_moins_1)
#print(V)
# Première tentative toujours haut sans etat terminaux : gamma = 0.9
# on obtient [ 0.03532384  0.519017    2.02384015  6.22181902 -0.01776443  1.95462675
#   4.01717154 -0.01868016  0.39768269  1.70035966  3.30263283]
# Deuxieme tentative avec etats terminaux, gamma = 1, [ 0.79889706  0.85514706  0.90514706  1.          0.74889706  0.54632353
#  -1.          0.69264706  0.64264706  0.5875     -0.86805556]
# pour gamma petit
#gamma = 0.01
#print(valeur_politique(pi,R,P,gamma))
# on obtient [-0.04040404 -0.04040297 -0.03934613  1.00904203 -0.04040404 -0.04134908
#  -0.99296197 -0.04040404 -0.04040406 -0.04041923 -0.04803215] qui est très proche de la récompense de chaque case

def verification_valeur(V,pi,R,P,gamma):
    nb_etat= P.shape[0]
    P_pi = np.zeros((nb_etat, nb_etat))
    R_pi = np.zeros(nb_etat)
    for i in range(nb_etat):
        P_pi[i, :] = P[i, pi[i], :]
        R_pi[i] = R[i, pi[i]]
        Vi = R_pi[i] + gamma * (P_pi@V)[i] # équation de bellman
        if not np.isclose(Vi,V[i]):
            raise ValueError(f"La valeur en {i} est fausse")
    print("tout est bon !")

#verification_valeur(V,pi,R,P,gamma)


# ---------------------- Apprentissage ----------------------

# But : Créer un agent passif sans prendre en paramètre la matrice de transition ni le vecteur de récompenses
# Donc il faut créer : - une fonction passer_etat_suivant qui nous donne l'état suivant et la récompense
#                      - une fonction agent_passif qui suit une politique pi sur la grille
#

def passer_etat_suivant(etat_actuel_input, pi_politique):
    nb_etat_env = P.shape[0]
    etat_suivant_idx_final = -1  # Initialisation
    etat_idx_pour_action_et_recompense = -1

    if isinstance(etat_actuel_input, (int,np.integer)):
        if 0 <= etat_actuel_input < nb_etat_env:
            etat_idx_pour_action_et_recompense = int(etat_actuel_input)
        else:
            raise ValueError(f"L'état (index) {etat_actuel_input} n'est pas valide.")
    elif isinstance(etat_actuel_input, (list, tuple)):
        etat_tuple_conv = tuple(etat_actuel_input) if isinstance(etat_actuel_input, list) else etat_actuel_input
        if len(etat_tuple_conv) != 2:
            raise ValueError(f"L'état (coordonnées) {etat_tuple_conv} doit avoir 2 éléments.")
        if etat_tuple_conv not in coordonnees:
            raise ValueError(f"L'état (coordonnées) {etat_tuple_conv} n'est pas sur la grille ou est un obstacle !")
        etat_idx_pour_action_et_recompense = etat_en_index[etat_tuple_conv]
    else:
        raise TypeError(f"Type d'état {type(etat_actuel_input)} non supporté.")

    action_choisie_idx = pi_politique[etat_idx_pour_action_et_recompense]
    etat_suivant_idx_final = np.random.choice(
        np.arange(nb_etat_env),
        p=P[etat_idx_pour_action_et_recompense, action_choisie_idx, :]
    )
    recompense_obtenue = R[etat_idx_pour_action_et_recompense, action_choisie_idx]

    return etat_suivant_idx_final, recompense_obtenue

#print(passer_etat_suivant(8,pi,P,R))
#print(passer_etat_suivant((1,1),pi,P,R)[0])
#print(passer_etat_suivant((1,2),pi,P,R)[0])
#print(passer_etat_suivant(10,pi,P,R)[0])
#print(passer_etat_suivant(123,pi,P,R)[0])

def agent_passif(V0_initial, pi_politique, P_transitions, R_recompenses,
                 coordonnees_env, etat_en_index_env,
                 etat_ini_idx=7, gamma=0.9, eps_erreur=1e-7, N_min_iter = 20, N_max_iter=10000):
    """
    V0_initial : vecteur initial des valeurs des états, de taille nb_etat
    pi_politique : la politique à suivre (liste/tableau d'indices d'action)
    P_transitions : Matrice P(s'|s,a)
    R_recompenses : Matrice R(s,a)
    coordonnees_env, etat_en_index_env : pour la fonction passer_etat_suivant
    etat_ini_idx : l'état initial en indice
    gamma : facteur d'actualisation
    eps_erreur : l'erreur tolérée pour la convergence
    N_max_iter : nombre d'itération maximal
    """
    nb_etat_env = P_transitions.shape[0]
    if len(V0_initial) != nb_etat_env:
        raise ValueError("V0_initial n'a pas la bonne taille.")

    nb_passages_etat = {i: 0 for i in range(nb_etat_env)}  # Nombre de passage par chaque état
    etat_actuel_idx = etat_ini_idx  # on commence avec l'état initial
    V_historique = [np.array(V0_initial, dtype=float)]  # Liste de tableaux NumPy contenant des flaot
    iter_count = 0

    while iter_count < N_max_iter:
        V_precedent = V_historique[-1]
        V_nouveau = V_precedent.copy()
        etat_a_mettre_a_jour_idx = etat_actuel_idx  # L'état actuel de l'agent est celui dont on met à jour la valeur.
        nb_passages_etat[etat_a_mettre_a_jour_idx] += 1
        #alpha = 1.0 / nb_passages_etat[etat_a_mettre_a_jour_idx]  # Taux d'apprentissage
        alpha = 60 / (59 + iter_count)
        etat_prochain_idx, recompense_transition = passer_etat_suivant(
            etat_actuel_idx, pi_politique, P_transitions, R_recompenses,
            coordonnees_env, etat_en_index_env )

        y = recompense_transition + gamma * V_precedent[etat_prochain_idx]
        V_nouveau[etat_a_mettre_a_jour_idx] = V_precedent[etat_a_mettre_a_jour_idx] + alpha * (y - V_precedent[etat_a_mettre_a_jour_idx])

        if np.linalg.norm(V_nouveau - V_precedent,ord=np.inf) < eps_erreur and iter_count > N_min_iter:
            print(f"Convergence atteinte à l'itération {iter_count + 1}.")
            return V_nouveau
        V_historique.append(V_nouveau)
        etat_actuel_idx = etat_prochain_idx
        iter_count += 1

    if iter_count >= N_max_iter:
        print(f"Nombre d'itérations maximal ({N_max_iter}) atteint sans convergence formelle.")
        return V_historique[-1], iter_count


def agent_adp_passif(
        pi_politique_non_terminal,
        fonction_env_pas_suivant,
        gamma,
        nb_etats_total,
        indices_etat_non_terminaux,
        indices_etat_terminaux,
        idx_etat_plus_1_global,
        idx_etat_moins_1_global,
        val_etat_plus_1_fixe=+1,
        val_etat_moins_1_fixe=-1,
        etat_initial_idx=7,
        nb_episodes_apprentissage=1000,
        nb_pas_max_par_episode=200
):

    nb_etats_non_terminaux = len(indices_etat_non_terminaux)

    if len(pi_politique_non_terminal) != nb_etats_non_terminaux:
        raise ValueError("Politique non terminale de mauvaise taille.")
    if etat_initial_idx in indices_etat_terminaux:
        raise ValueError("L'état initial ne peut pas être terminal.")


    N_s_vers_s_prime = np.zeros((nb_etats_total, nb_etats_total))
    N_s_departs = np.zeros(nb_etats_total)
    R_pi = np.zeros(nb_etats_total)

    pi_complete = np.zeros(nb_etats_total, dtype=int)
    index_etat_local = {global_idx: i for i, global_idx in enumerate(indices_etat_non_terminaux)}
    for s_global_idx in indices_etat_non_terminaux:
        pi_complete[s_global_idx] = pi_politique_non_terminal[index_etat_local[s_global_idx]]

    for episode in range(nb_episodes_apprentissage):
        etat_s_actuel = random.choice(indices_etat_non_terminaux)
        for etat in indices_etat_non_terminaux:
            if N_s_departs[etat] == 0:
                etat_s_actuel = etat
        for num_pas in range(nb_pas_max_par_episode):
            if etat_s_actuel in indices_etat_terminaux:
                break

            etat_s_prime_suivant_global, r_pi = fonction_env_pas_suivant(etat_s_actuel,
                pi_complete)
            R_pi[etat_s_prime_suivant_global] = r_pi
            N_s_vers_s_prime[etat_s_actuel, etat_s_prime_suivant_global] += 1
            N_s_departs[etat_s_actuel] += 1

            etat_s_actuel = etat_s_prime_suivant_global

    P_pi_appris_global = np.zeros((nb_etats_total, nb_etats_total))
    for s_idx in range(nb_etats_total):
        if N_s_departs[s_idx] > 0:
            P_pi_appris_global[s_idx, :] = N_s_vers_s_prime[s_idx, :] / N_s_departs[s_idx]
        elif s_idx in indices_etat_terminaux:
            P_pi_appris_global[s_idx, s_idx] = 1.0
        else:
            P_pi_appris_global[s_idx, s_idx] = 1.0


    P_pi_non_term_solve = np.zeros((nb_etats_non_terminaux, nb_etats_non_terminaux))

    for i_local_s, s_global_idx in enumerate(indices_etat_non_terminaux):
        if idx_etat_plus_1_global is not None:
            R_pi[i_local_s] += gamma * P_pi_appris_global[
                s_global_idx, idx_etat_plus_1_global] * val_etat_plus_1_fixe
        if idx_etat_moins_1_global is not None:
            R_pi[i_local_s] += gamma * P_pi_appris_global[
                s_global_idx, idx_etat_moins_1_global] * val_etat_moins_1_fixe

        for j_local_s_prime, s_prime_global_idx in enumerate(indices_etat_non_terminaux):
            P_pi_non_term_solve[i_local_s, j_local_s_prime] = P_pi_appris_global[s_global_idx, s_prime_global_idx]

    I_non_term = np.eye(nb_etats_non_terminaux)
    R = R_pi[0:nb_etats_non_terminaux]
    V_non_terminaux_appris = np.linalg.solve(I_non_term - gamma * P_pi_non_term_solve, R)

    # Reconstruire le vecteur V complet
    V_complet_final = np.zeros(nb_etats_total)
    V_complet_final[idx_etat_plus_1_global] = val_etat_plus_1_fixe
    V_complet_final[idx_etat_moins_1_global] = val_etat_moins_1_fixe
    for i_local_fin, s_global_idx_fin in enumerate(indices_etat_non_terminaux):
        V_complet_final[s_global_idx_fin] = V_non_terminaux_appris[i_local_fin]

    return V_complet_final


def agent_passif_terminal(
        V0_initial,  # Vecteur initial des valeurs (pour tous les états, 11)
        politique_non_terminal,  # Politique pour les états NON TERMINAUX (taille 9)
        fonction_pas_suivant,  # La fonction qui simule l'environnement
        nb_etats_total_env,  # Doit être connu (ici 11)
        indices_etats_non_terminaux,  # Liste des indices des états non terminaux
        indices_etats_terminaux,  # Liste des indices globaux [idx_plus_1, idx_moins_1]
        etat_ini_idx = 7,  # État initial (doit être un index d'un état non terminal)
        gamma=0.9,
        eps_erreur=1e-5,
        N_min=100,
        N_max_iter=20000,
        val_etat_plus_1=+1.0,  # Valeur de l'état terminal +1
        val_etat_moins_1=-1.0,  # Valeur de l'état terminal -1
):
    """ agent td passif mais avec etats terminaux """

    if len(V0_initial) != nb_etats_total_env:
        raise ValueError(f"V0_initial n'a pas la bonne taille. Sa taille doit être {nb_etats_total_env} tandis que le vecteur donné est de taille {len(V0_initial)}")
    if len(politique_non_terminal) != len(indices_etats_non_terminaux):
        raise ValueError(f"pi_politique_non_terminal n'a pas la bonne taille. Sa taille doit être {len(indices_etats_non_terminaux)} tandis que le vecteur donné est de taille {len(politique_non_terminal)}")
    if etat_ini_idx in indices_etats_terminaux:
        raise ValueError("L'état initial ne peut pas être un état terminal.")

    nb_passages_etat = {i: 0 for i in indices_etats_non_terminaux}
    etat_actuel_idx = etat_ini_idx # etat actuel est l'etat initial

    idx_etat_plus_1 = indices_etats_terminaux[0]
    idx_etat_moins_1 = indices_etats_terminaux[1]
    V_estime = np.array(V0_initial, dtype=float) # Valeur estimée
    V_estime[idx_etat_plus_1] = val_etat_plus_1
    V_estime[idx_etat_moins_1] = val_etat_moins_1
    V_historique = []
    V_historique.append(V_estime)

    politique_env = np.zeros(nb_etats_total_env, dtype=int)  # la politique pour tout l'env, les actions dans les
    # cases terminaux est 0 mais elles ne sont pas prises en compte dans l'algo
    for idx_list, idx_etat in enumerate(indices_etats_non_terminaux):
        politique_env[idx_etat] = politique_non_terminal[idx_list]

    for iter_count in range(N_max_iter):
        V_k = V_estime.copy()

        if etat_actuel_idx in indices_etats_terminaux:
            etat_actuel_idx = etat_ini_idx # on re-initialise l agent
            continue # on modifie pas les valeurs des etats terminaux

        etat_passe_idx = etat_actuel_idx
        nb_passages_etat[etat_passe_idx] += 1
        alpha = 60 / (59 + iter_count)
        #alpha = 1.0 / nb_passages_etat[etat_passe_idx]

        # L'agent passe à l etat suivant
        etat_prochain_global_idx, recompense_transition = fonction_pas_suivant(
            etat_actuel_idx,
            politique_env
        )

        y_k = recompense_transition + gamma * V_estime[etat_prochain_global_idx] # y représentant la récompense moyenne de cet état
        surprise = y_k - V_estime[etat_passe_idx]
        V_estime[etat_passe_idx] = V_estime[etat_passe_idx] + alpha * surprise # difference temporel

        V_non_term_actuel = V_estime[indices_etats_non_terminaux]
        V_non_term_precedent = V_k[indices_etats_non_terminaux]
        norm = np.linalg.norm(V_non_term_actuel - V_non_term_precedent, ord=np.inf)

        if norm < eps_erreur and iter_count > N_min:
            print(f"Convergence atteinte à l'itération {iter_count + 1}.")
            V_estime[idx_etat_plus_1] = val_etat_plus_1
            V_estime[idx_etat_moins_1] = val_etat_moins_1
            return V_estime, iter_count + 1

        etat_actuel_idx = etat_prochain_global_idx

    print(f"Nombre d'itérations maximal ({N_max_iter}) atteint.")
    V_estime[idx_etat_plus_1] = val_etat_plus_1
    V_estime[idx_etat_moins_1] = val_etat_moins_1
    return V_estime, N_max_iter


def agent_actif_Q(
        fonction_pas_suivant,
        nb_etats_total,
        nb_actions_total,
        indices_etats_terminaux,
        indices_etats_non_terminaux,
        gamma=1.0,
        epsilon=0.3,
        nb_episodes=50000
):
    """
    agent td actif Q learning
    retourne un tableau Q(s,a)
    """
    Q = np.zeros((nb_etats_total, nb_actions_total))
    N_sa = np.zeros((nb_etats_total, nb_actions_total))
    indices_terminaux_set = set(indices_etats_terminaux)

    for i in range(nb_episodes):
        etat_actuel_idx = random.choice(indices_etats_non_terminaux)

        done = False
        while not done:
            # Choix d'une action
            if np.random.uniform(0, 1) < epsilon:
                action_choisie_idx = random.randint(0, nb_actions_total - 1)
            else:
                action_choisie_idx = np.argmax(Q[etat_actuel_idx])

            etat_prochain_idx, recompense = fonction_pas_suivant(etat_actuel_idx, action_choisie_idx)
            done = etat_prochain_idx in indices_terminaux_set

            N_sa[etat_actuel_idx, action_choisie_idx] += 1
            alpha = 60 / (59 + N_sa[etat_actuel_idx, action_choisie_idx])

            valeur_q_actuelle = Q[etat_actuel_idx, action_choisie_idx]
            valeur_max_q_future = 0
            if not done:
                valeur_max_q_future = np.max(Q[etat_prochain_idx])

            cible = recompense + gamma * valeur_max_q_future

            Q[etat_actuel_idx, action_choisie_idx] = valeur_q_actuelle + alpha * (cible - valeur_q_actuelle)
            etat_actuel_idx = etat_prochain_idx
    return Q


def valeur_optimal_iteration(P_global, gamma, R,
                             indices_etats_non_terminaux,
                             idx_etat_plus_1,
                             idx_etat_moins_1,
                             epsilon=1e-5,
                             max_iter=1000,
                             val_etat_plus_1=+1.0,
                             val_etat_moins_1=-1.0):
    """
    Calcule les valeurs optimales des états V*(s) en utilisant Value Iteration.
    R est la récompense pour être dans un état non terminal .
    """
    nb_etats_total = P_global.shape[0]
    nb_actions_total = P_global.shape[1]

    V_k = np.zeros(nb_etats_total)
    V_k[idx_etat_plus_1] = val_etat_plus_1
    V_k[idx_etat_moins_1] = val_etat_moins_1

    politique_idx= {i:j for j,i in enumerate(indices_etats_non_terminaux)}
    politique_optimale = {i:0 for i in range(len(indices_etats_non_terminaux))}

    for iteration in range(max_iter):
        V_k_plus_1 = V_k.copy()  # on commence avec les valeurs de Vk
        delta = 0

        for s_idx_global in indices_etats_non_terminaux:  # on tourne que sur les etats non terminaux
            valeurs_s = np.zeros(nb_actions_total)  # Q(s,a) pour l' etat s

            for a_idx in range(nb_actions_total):
                somme = 0
                for s_prime in range(nb_etats_total):
                    somme += P_global[s_idx_global, a_idx, s_prime] * V_k[s_prime]
                valeurs_s[a_idx] = R + gamma * somme # equation de bellman

            # V_k+1(s) = max_a Q(s,a)
            max_valeur_s = np.max(valeurs_s) # on prend la plus grande valeur pour l'etat s
            meilleur_action_s = int(np.argmax(valeurs_s))
            politique_optimale[politique_idx[s_idx_global]] = meilleur_action_s
            delta = max(delta, abs(max_valeur_s - V_k[s_idx_global])) # on calcule l'ecart entre Vk+1 (s) et Vk(s) et on prend le max pour tous les s
            V_k_plus_1[s_idx_global] = max_valeur_s

        V_k = V_k_plus_1  # on met à jour V_k pour la prochaine iteration

        if delta < epsilon:
            print(f"L'algorithme d'Itération de la Valeur a convergé à l'iteration {iteration + 1}")
            break
    else:
        print(f"L'algorithme d'Itération de la Valeur n'a pas convergé après {max_iter} iterations.")

    return V_k, politique_optimale


def agent_passif_pour_graphiques(V0_initial, politique_non_terminal,
                                 fonction_pas_suivant, nb_etats_total_env, indices_etats_non_terminaux,
                                 indices_etats_terminaux, val_etat_plus_1=+1.0, val_etat_moins_1=-1.0,
                                 etat_ini_idx=7, gamma=1.0, nombre_episodes_max=500):

    if len(V0_initial) != nb_etats_total_env:
        raise ValueError(f"V0_initial n'a pas la bonne taille. Sa taille doit être {nb_etats_total_env} tandis que le vecteur donné est de taille {len(V0_initial)}")
    if len(politique_non_terminal) != len(indices_etats_non_terminaux):
        raise ValueError(f"pi_politique_non_terminal n'a pas la bonne taille. Sa taille doit être {len(indices_etats_non_terminaux)} tandis que le vecteur donné est de taille {len(politique_non_terminal)}")
    if etat_ini_idx in indices_etats_terminaux:
        raise ValueError("L'état initial ne peut pas être un état terminal.")

    nb_passages_etat = {i: 0 for i in indices_etats_non_terminaux}

    V_estime = np.array(V0_initial, dtype=float)
    idx_etat_plus_1 = indices_etats_terminaux[0]
    idx_etat_moins_1 = indices_etats_terminaux[1]
    V_estime[idx_etat_plus_1] = val_etat_plus_1
    V_estime[idx_etat_moins_1] = val_etat_moins_1

    historique_V = []  # Pour stocker V_estime à la fin de chaque trial

    politique_env = np.zeros(nb_etats_total_env, dtype=int)
    for idx_list, idx_etat in enumerate(indices_etats_non_terminaux):
        politique_env[idx_etat] = politique_non_terminal[idx_list]

    etat_actuel_idx = etat_ini_idx

    for episode in range(nombre_episodes_max):

        # On réinitialise l'état au début de chaque trial/ épisode
        if etat_actuel_idx in indices_etats_terminaux:
            etat_actuel_idx = etat_ini_idx

        # Simuler un épisode jusqu'à atteindre un état terminal
        pas_dans_episode = 0
        max_pas_par_episode = 200  # Pour éviter les épisodes infinis si la politique est mauvaise

        while etat_actuel_idx not in indices_etats_terminaux and pas_dans_episode < max_pas_par_episode :
            etat_passe_idx = etat_actuel_idx  # État où l'agent était

            nb_passages_etat[etat_passe_idx] += 1
            alpha = 60 / (59 + nb_passages_etat[etat_passe_idx] )

            etat_prochain_global_idx, recompense_transition = fonction_pas_suivant(
                etat_actuel_idx,
                politique_env)
            y = recompense_transition + gamma * V_estime[etat_prochain_global_idx]
            V_estime[etat_passe_idx] = V_estime[etat_passe_idx] + alpha * (y - V_estime[etat_passe_idx])

            etat_actuel_idx = etat_prochain_global_idx
            pas_dans_episode += 1

        historique_V.append(V_estime.copy())

    return np.array(historique_V)  # Tableau de forme (nombre_episodes_max, nb_etats_total_env)

def passer_etat_suivant_action_explicite(etat_actuel_idx_global, action_choisie_idx):
    """
    Simule un pas dans l'environnement en utilisant une action explicite.
    Retourne (etat_suivant_idx, recompense_R_sa).
    """
    etat_suivant_idx = np.random.choice(
        np.arange(nb_etat),
        p=P[etat_actuel_idx_global, action_choisie_idx, :]
    )
    recompense_obtenue = R[etat_actuel_idx_global, action_choisie_idx]
    return etat_suivant_idx, recompense_obtenue



def agent_adp_passif_graphiques(
        pi_politique_non_terminal,
        fonction_env_pas_suivant,
        gamma,
        nb_etats_total,
        indices_etat_non_terminaux_liste,
        indices_etat_terminaux_liste,
        idx_etat_plus_1,
        val_etat_plus_1,
        idx_etat_moins_1,
        val_etat_moins_1,
        etat_initial_idx,
        nombre_episodes_apprentissage=100,
        nb_pas_max_par_episode=200
):
    nb_etats_non_terminaux_count = len(indices_etat_non_terminaux_liste)

    if len(pi_politique_non_terminal) != nb_etats_non_terminaux_count:
        raise ValueError("Politique non terminale de mauvaise taille.")
    if etat_initial_idx in indices_etat_terminaux_liste:
        raise ValueError("L'état initial ne peut pas être terminal.")

    # Modèle appris par l'agent ADP
    N_s_vers_s_prime_comptes = np.zeros((nb_etats_total, nb_etats_total))
    N_s_departs_comptes = np.zeros(nb_etats_total)
    R_pi_appris_par_s_depart = np.full(nb_etats_total, 0.0)
    for idx_term in indices_etat_terminaux_liste:
        R_pi_appris_par_s_depart[idx_term] = 0.0

    pi_complete = np.zeros(nb_etats_total, dtype=int)
    map_global_nt_to_local = {s_g: i_l for i_l, s_g in enumerate(indices_etat_non_terminaux_liste)}
    for s_global_map_idx in indices_etat_non_terminaux_liste:
        pi_complete[s_global_map_idx] = pi_politique_non_terminal[map_global_nt_to_local[s_global_map_idx]]

    historique_V_adp_final = []
    etat_s_courant_episode = etat_initial_idx

    # --- Phase d'exploration initiale optionnelle (quelques épisodes pour peupler les comptes) ---

    nb_episodes_pre_exploration = nb_etats_non_terminaux_count  # visiter chaque état non-term 1 fois
    temp_etat_pre_explo = etat_initial_idx
    for _ in range(nb_episodes_pre_exploration):  # Ce sont des PAS, pas des épisodes complets
        if temp_etat_pre_explo in indices_etat_terminaux_liste:
            temp_etat_pre_explo = random.choice(indices_etat_non_terminaux_liste)  # Réinitialiser si terminal

        action_pre_explo = pi_complete[temp_etat_pre_explo]
        s_prime_pre_explo, r_sa_pre_explo = fonction_env_pas_suivant(
            temp_etat_pre_explo,
            action_pre_explo
        )
        N_s_vers_s_prime_comptes[temp_etat_pre_explo, s_prime_pre_explo] += 1
        N_s_departs_comptes[temp_etat_pre_explo] += 1
        R_pi_appris_par_s_depart[temp_etat_pre_explo] = r_sa_pre_explo
        temp_etat_pre_explo = s_prime_pre_explo

    # --- Phase des trials ---
    for episode_num in range(nombre_episodes_apprentissage):
        if episode_num > 0 or N_s_departs_comptes[
            etat_s_courant_episode] > 0:
            etat_s_courant_episode = random.choice(indices_etat_non_terminaux_liste)

        for _ in range(nb_pas_max_par_episode):
            if etat_s_courant_episode in indices_etat_terminaux_liste:
                break

            action_a_faire = pi_complete[etat_s_courant_episode]
            etat_s_prime_suivant_global, r_sa_obtenue = fonction_env_pas_suivant(
                etat_s_courant_episode,
                action_a_faire
            )
            N_s_vers_s_prime_comptes[etat_s_courant_episode, etat_s_prime_suivant_global] += 1
            N_s_departs_comptes[etat_s_courant_episode] += 1
            R_pi_appris_par_s_depart[etat_s_courant_episode] = r_sa_obtenue
            etat_s_courant_episode = etat_s_prime_suivant_global

        # --- Étape de POLICY-EVALUATION ---
        P_pi_appris_global_ce_trial = np.zeros((nb_etats_total, nb_etats_total))
        for s_norm_idx in range(nb_etats_total):
            if N_s_departs_comptes[s_norm_idx] > 0:
                P_pi_appris_global_ce_trial[s_norm_idx, :] = N_s_vers_s_prime_comptes[s_norm_idx, :] / \
                                                             N_s_departs_comptes[s_norm_idx]
            elif s_norm_idx in indices_etat_terminaux_liste:
                P_pi_appris_global_ce_trial[s_norm_idx, s_norm_idx] = 1.0
            else:
                P_pi_appris_global_ce_trial[s_norm_idx, s_norm_idx] = 1.0

        P_pi_non_term_pour_solve = np.zeros((nb_etats_non_terminaux_count, nb_etats_non_terminaux_count))
        R_pi_non_term_modifie_pour_solve = np.zeros(nb_etats_non_terminaux_count)

        for i_local_s_eval, s_global_idx_eval in enumerate(indices_etat_non_terminaux_liste):
            R_pi_non_term_modifie_pour_solve[i_local_s_eval] = R_pi_appris_par_s_depart[s_global_idx_eval]
            if idx_etat_plus_1 is not None:
                R_pi_non_term_modifie_pour_solve[i_local_s_eval] += gamma * P_pi_appris_global_ce_trial[
                    s_global_idx_eval, idx_etat_plus_1] * val_etat_plus_1
            if idx_etat_moins_1 is not None:
                R_pi_non_term_modifie_pour_solve[i_local_s_eval] += gamma * P_pi_appris_global_ce_trial[
                    s_global_idx_eval, idx_etat_moins_1] * val_etat_moins_1
            for j_local_s_prime_eval, s_prime_global_idx_eval in enumerate(indices_etat_non_terminaux_liste):
                P_pi_non_term_pour_solve[i_local_s_eval, j_local_s_prime_eval] = P_pi_appris_global_ce_trial[
                    s_global_idx_eval, s_prime_global_idx_eval]

        I_non_term_eval = np.eye(nb_etats_non_terminaux_count)
        V_non_terminaux_appris_ce_trial = np.zeros(nb_etats_non_terminaux_count)
        mat_a_inverser_eval = (I_non_term_eval - gamma * P_pi_non_term_pour_solve)

        try:
            # Pour gamma=1, on vérifie si la matrice est singulière
            if np.isclose(gamma, 1.0) and (
                    np.linalg.cond(mat_a_inverser_eval) > 1  or np.isclose(
                    np.linalg.det(mat_a_inverser_eval), 0)):
                V_non_terminaux_appris_ce_trial = \
                np.linalg.lstsq(mat_a_inverser_eval, R_pi_non_term_modifie_pour_solve, rcond=None)[0]
            else:
                V_non_terminaux_appris_ce_trial = np.linalg.solve(mat_a_inverser_eval, R_pi_non_term_modifie_pour_solve)
        except np.linalg.LinAlgError:
            V_non_terminaux_appris_ce_trial = \
            np.linalg.lstsq(mat_a_inverser_eval, R_pi_non_term_modifie_pour_solve, rcond=None)[0]

        V_complet_ce_trial = np.zeros(nb_etats_total)
        V_complet_ce_trial[idx_etat_plus_1] = val_etat_plus_1
        V_complet_ce_trial[idx_etat_moins_1] = val_etat_moins_1
        for i_local_fin_adp, s_global_idx_fin_adp in enumerate(indices_etat_non_terminaux_liste):
            V_complet_ce_trial[s_global_idx_fin_adp] = V_non_terminaux_appris_ce_trial[i_local_fin_adp]
        historique_V_adp_final.append(V_complet_ce_trial.copy())

    return np.array(historique_V_adp_final)


# ------- Graphiques et tests --------
gamma = 1.0
R = -0.04

V_optimal, pi_opti = valeur_optimal_iteration(
    P, gamma, R,
    indices_etats_non_terminaux,
    idx_etat_plus_1,
    idx_etat_moins_1,
    epsilon=1e-5)

print("\nValeurs optimales V* calculées par itération de valeur (gamma=1.0):")
valeurs_grille_optimal = np.full((lignes_grille, cols_grille), np.nan)
for idx in range(nb_etat):
    r_vi, c_vi = index_en_etat[idx]
    valeurs_grille_optimal[r_vi, c_vi] = V_optimal[idx]

print(np.round(valeurs_grille_optimal, 3))

print("\nValeur de la politique optimale avec gamma = 1")

R = np.full((nb_etat, nb_action), -0.04)
for s_term_idx in indices_etats_terminaux:
    R[s_term_idx, :] = 0.0

gamma = 1
V = valeur_politique_livre(pi_opti,R,P,gamma,indices_etats_non_terminaux,
    idx_etat_plus_1, val_etat_plus_1,
    idx_etat_moins_1, val_etat_moins_1)

valeurs_grille_optimal = np.full((lignes_grille, cols_grille), np.nan)
for idx in range(nb_etat): # nb_etat est 11
    r_vi, c_vi = index_en_etat[idx]
    valeurs_grille_optimal[r_vi, c_vi] = V[idx]

print(np.round(valeurs_grille_optimal, 3))

grille_politique_fleches = np.full((lignes_grille, cols_grille), " ", dtype=object)

grille_politique_fleches[index_en_etat[idx_etat_plus_1]] = "+1"
grille_politique_fleches[index_en_etat[idx_etat_moins_1]] = "-1"
grille_politique_fleches[obstacle_coord] = "MUR"
action_idx_en_signe = {0:"↑", 1:"↓", 2:"←",3:"→"}

for local_idx, global_s_idx in enumerate(indices_etats_non_terminaux):

    action_optimal_pour_s = pi_opti[local_idx]
    r, c = index_en_etat[global_s_idx]
    grille_politique_fleches[r, c] = action_idx_en_signe[action_optimal_pour_s]

print("\nPolitique Optimale (flèches):")
for r_idx in range(lignes_grille):
    r_str = "| "
    for c_idx in range(cols_grille):
        cell_contenu = grille_politique_fleches[r_idx, c_idx]
        if cell_contenu == "+1" or cell_contenu == "-1":
            r_str += f"{cell_contenu:^4} | "
        elif cell_contenu == "MUR":
             r_str += f"{cell_contenu:^4} | "
        else:
            r_str += f" {cell_contenu}   | "
    print(r_str)
    if r_idx < lignes_grille - 1:
        print("-" * len(r_str))

V0 = [0]*11

V_agent_passif, nb_iter_passif = agent_passif_terminal(V0, pi_opti, passer_etat_suivant,
                                                       nb_etat, indices_etats_non_terminaux,
                                                       indices_etats_terminaux,7,gamma = 1.0,
                                                       N_min = 10000)

print(f"\nValeur des états avec la méthode Agent passif pour la politique optimal ({nb_iter_passif} iter):  ")
valeur_grille_passif = np.full((lignes_grille,cols_grille), np.nan)
for idx_list,idx_etat in enumerate(coordonnees):
    valeur_grille_passif[idx_etat] = V_agent_passif[idx_list]

print(valeur_grille_passif)

print(f"\nValeur des états avec la méthode Agent ADP passif pour la politique optimal:  ")

V_passif_adp = agent_adp_passif(pi_opti,passer_etat_suivant,gamma,nb_etat,
                                indices_etats_non_terminaux,indices_etats_terminaux,
                                idx_etat_plus_1,idx_etat_moins_1,+1,-1,7,1000)

for idx_list,idx_etat in enumerate(coordonnees):
    valeur_grille_passif[idx_etat] = V_passif_adp[idx_list]

print(valeur_grille_passif)


print("\nGraphique des utilités : ")
nb_trials = 500
etat_initial = 7
gamma_agent_passif = 1.0

historique_V_passif = agent_passif_pour_graphiques(V0, pi_opti, passer_etat_suivant,
                                                       nb_etat, indices_etats_non_terminaux,
                                                       indices_etats_terminaux)

etats_pour_graphique = {etat_en_index[(2,0)]: "(1,1)",
                        etat_en_index[(0,0)]: "(1,3)",
                        etat_en_index[(2,1)]: "(2,1)",
                        etat_en_index[(0,2)]: "(3,3)",
                        idx_etat_plus_1:       "(4,3)"}

indices_graph_a = list(etats_pour_graphique.keys())

trials_axe_x = np.arange(1, nb_trials + 1)

plt.figure(figsize=(10, 6))

styles_ligne = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]

for i, s_idx_a in enumerate(indices_graph_a):

    label_graph = etats_pour_graphique[s_idx_a]
    style = styles_ligne[i]
    plt.plot(trials_axe_x, historique_V_passif[:, s_idx_a],
                 label=label_graph, linestyle=style, linewidth=0.8)

plt.xlabel("Nombre d'épisodes (trials)")
plt.ylabel("Estimations d'utilité V(s)")
plt.title(f"Courbes d'apprentissage TD")
plt.legend(loc='center right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.ylim(-0.2, 1.1)
plt.xlim(0, nb_trials)
plt.show()

# --- RMS TD ---
nombre_runs_rms = 20
nombre_trials_pour_rms = 100
gamma_rms = 1.0
etat_specifique_rms_idx = etat_en_index[(2, 0)]
valeur_vraie_etat_specifique = V_optimal[etat_specifique_rms_idx]

toutes_erreurs_abs_etat_specifique_par_run = np.zeros((nombre_runs_rms, nombre_trials_pour_rms))

for run_idx_rms in range(nombre_runs_rms):

    historique_V_ce_run_rms = agent_passif_pour_graphiques(
        V0_initial=V0.copy(),
        politique_non_terminal=pi_opti,
        fonction_pas_suivant=passer_etat_suivant,
        nb_etats_total_env=nb_etat,
        indices_etats_non_terminaux=indices_etats_non_terminaux,
        indices_etats_terminaux=indices_etats_terminaux,
        gamma=gamma_rms,
        nombre_episodes_max=nombre_trials_pour_rms
    )
    # historique_V_ce_run_rms a la forme (nombre_trials_pour_rms, nb_etat)

    for trial_idx_rms in range(nombre_trials_pour_rms):
        V_estime_ce_trial_etat_specifique = historique_V_ce_run_rms[trial_idx_rms, etat_specifique_rms_idx]
        erreur = np.square(V_estime_ce_trial_etat_specifique - valeur_vraie_etat_specifique)
        toutes_erreurs_abs_etat_specifique_par_run[run_idx_rms, trial_idx_rms] = erreur

erreur_quadratique_moyenne_etat_specifique = np.mean(toutes_erreurs_abs_etat_specifique_par_run, axis=0)

trials_axe_x_fig_b_final = np.arange(1, nombre_trials_pour_rms + 1)

plt.figure(figsize=(7, 5))
plt.plot(trials_axe_x_fig_b_final, erreur_quadratique_moyenne_etat_specifique, linewidth=1.0)
plt.xlabel("Nombre d'épisodes (trials)")
plt.ylabel(f"Erreur RMS (absolue) pour U({coordonnees[etat_specifique_rms_idx]}) (moy. sur runs)")
plt.title(f"Erreur pour U(1,1) (moy. sur {nombre_runs_rms} runs, {nombre_trials_pour_rms} premiers trials)")
plt.grid(True, linestyle=':', alpha=0.7)
ylim_max_rms = max(0.6, np.max(erreur_quadratique_moyenne_etat_specifique) * 1.1 if len(
    erreur_quadratique_moyenne_etat_specifique) > 0 else 0.6)
plt.ylim(0, ylim_max_rms)
plt.xlim(0, nombre_trials_pour_rms)
plt.show()

# --- Section d'Exécution pour les Graphiques ADP ---
print("\n--- Génération des données pour les graphiques de l'Agent ADP Passif ---")

nb_trials_adp = 100
etat_initial = 7
gamma_agent_passif_adp = 1.0

historique_V_passif_adp = agent_adp_passif_graphiques(pi_opti,passer_etat_suivant_action_explicite,gamma_agent_passif_adp,
                                                      nb_etat,indices_etats_non_terminaux,indices_etats_terminaux,
                                                      idx_etat_plus_1,val_etat_plus_1,idx_etat_moins_1,val_etat_moins_1,etat_initial,
                                                      nb_trials_adp)

etats_pour_graphique = {etat_en_index[(2,0)]: "(1,1)",
                        etat_en_index[(0,0)]: "(1,3)",
                        etat_en_index[(2,1)]: "(2,1)",
                        etat_en_index[(0,2)]: "(3,3)",
                        idx_etat_plus_1:       "(4,3)"}

indices_graph_a = list(etats_pour_graphique.keys())

trials_axe_x = np.arange(1, nb_trials_adp + 1)

plt.figure(figsize=(10, 6))

styles_ligne = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]

for i, s_idx_a in enumerate(indices_graph_a):

    label_graph = etats_pour_graphique[s_idx_a]
    style = styles_ligne[i]
    plt.plot(trials_axe_x, historique_V_passif_adp[:, s_idx_a],
                 label=label_graph, linestyle=style, linewidth=0.8)

plt.xlabel("Nombre d'épisodes (trials)")
plt.ylabel("Estimations d'utilité V(s)")
plt.title(f"Courbes d'apprentissage ADP")
plt.legend(loc='center right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.ylim(-0.2, 1.1)
plt.xlim(0, nb_trials_adp)
plt.show()

# --- RMS ADP ---
nombre_runs_rms = 20
nombre_trials_pour_rms = 100
gamma_rms = 1.0
etat_specifique_rms_idx = etat_en_index[(2, 0)]
valeur_vraie_etat_specifique = V_optimal[etat_specifique_rms_idx]

toutes_erreurs_abs_etat_specifique_par_run = np.zeros((nombre_runs_rms, nombre_trials_pour_rms))

for run_idx_rms in range(nombre_runs_rms):

    historique_V_ce_run_rms_adp = agent_adp_passif_graphiques(pi_opti,passer_etat_suivant_action_explicite,gamma_agent_passif_adp,
                                                      nb_etat,indices_etats_non_terminaux,indices_etats_terminaux,
                                                      idx_etat_plus_1,val_etat_plus_1,idx_etat_moins_1,val_etat_moins_1,etat_initial,
                                                      nb_trials_adp)

    for trial_idx_rms in range(nombre_trials_pour_rms):
        V_estime_ce_trial_etat_specifique_adp = historique_V_ce_run_rms_adp[trial_idx_rms, etat_specifique_rms_idx]
        erreur = np.square(V_estime_ce_trial_etat_specifique_adp - valeur_vraie_etat_specifique)
        toutes_erreurs_abs_etat_specifique_par_run[run_idx_rms, trial_idx_rms] = erreur

erreur_quadratique_moyenne_etat_specifique_adp = np.mean(toutes_erreurs_abs_etat_specifique_par_run, axis=0)

trials_axe_x_fig_b_final = np.arange(1, nombre_trials_pour_rms + 1)

plt.figure(figsize=(7, 5))
plt.plot(trials_axe_x_fig_b_final, erreur_quadratique_moyenne_etat_specifique_adp, linewidth=1.0)
plt.xlabel("Nombre d'épisodes (trials)")
plt.ylabel(f"Erreur RMS (absolue) pour U({coordonnees[etat_specifique_rms_idx]}) (moy. sur runs)")
plt.title(f"Erreur pour U(1,1) (moy. sur {nombre_runs_rms} runs, {nombre_trials_pour_rms} premiers trials)")
plt.grid(True, linestyle=':', alpha=0.7)
ylim_max_rms = max(0.6, np.max(erreur_quadratique_moyenne_etat_specifique_adp) * 1.1 if len(
    erreur_quadratique_moyenne_etat_specifique_adp) > 0 else 0.6)
plt.ylim(0, ylim_max_rms)
plt.xlim(0, nombre_trials_pour_rms)
plt.show()


print("\n--- Lancement de l'agent Q-Learning ---")

R_test_td = np.full((nb_etat, nb_action), -0.04)

def passer_etat_suivant_R_zero(etat_actuel_idx_global, action_choisie_idx):
    etat_suivant_idx = np.random.choice(
        np.arange(nb_etat),
        p=P[etat_actuel_idx_global, action_choisie_idx, :]
    )
    recompense_obtenue = R_test_td[etat_actuel_idx_global, action_choisie_idx]
    if etat_suivant_idx == idx_etat_plus_1:
        recompense_obtenue = val_etat_plus_1
    elif etat_suivant_idx == idx_etat_moins_1:
        recompense_obtenue = val_etat_moins_1

    return etat_suivant_idx, recompense_obtenue

Q_td_appris = agent_actif_Q(
    fonction_pas_suivant=passer_etat_suivant_R_zero,
    nb_etats_total=nb_etat,
    nb_actions_total=nb_action,
    indices_etats_terminaux=indices_etats_terminaux,
    indices_etats_non_terminaux=indices_etats_non_terminaux,
    gamma=1.0,
    epsilon=0.3,
    nb_episodes=20000
)

V_td_learning = np.max(Q_td_appris, axis=1)
V_td_learning[idx_etat_plus_1] = val_etat_plus_1
V_td_learning[idx_etat_moins_1] = val_etat_moins_1
pi_td_learning = np.argmax(Q_td_appris, axis=1)

print("\nValeurs des états V* apprises par le Q-Learning TD:")
valeurs_grille_td = np.full((lignes_grille, cols_grille), np.nan)
for idx in range(nb_etat):
    r_td, c_td = index_en_etat[idx]
    valeurs_grille_td[r_td, c_td] = V_td_learning[idx]

print(np.round(valeurs_grille_td, 3))

print("\nPolitique Optimale π* apprise par le Q-Learning TD:")
grille_politique_td = np.full((lignes_grille, cols_grille), " ", dtype=object)
grille_politique_td[index_en_etat[idx_etat_plus_1]] = "+1"
grille_politique_td[index_en_etat[idx_etat_moins_1]] = "-1"
grille_politique_td[obstacle_coord] = "MUR"

for s_idx_global in indices_etats_non_terminaux:
    action_optimale_td = pi_td_learning[s_idx_global]
    r, c = index_en_etat[s_idx_global]
    grille_politique_td[r, c] = action_idx_en_signe[action_optimale_td]

for r_idx in range(lignes_grille):
    r_str = "| "
    for c_idx in range(cols_grille):
        cell_contenu = grille_politique_td[r_idx, c_idx]
        if len(str(cell_contenu)) > 1:
            r_str += f"{cell_contenu:^4} | "
        else:
            r_str += f" {cell_contenu}   | "
    print(r_str)
    if r_idx < lignes_grille - 1:
        print("-" * len(r_str))

print(Q_td_appris)

# ---------------------- Temps de Calcul ---------------------------

def creer_environnement_grille(
        lignes: int,
        cols: int,
        recompense_mouvement: float = -0.04,
        etats_terminaux: dict = None,  # ex: {(0, 3): 1.0, (1, 3): -1.0}
        obstacles: list = None,  # ex: [(1, 1)]
        prob_intention: float = 0.8
):
    """
    Crée un environnement de grille MDP complet.

    Args:
        lignes (int): Nombre de lignes de la grille.
        cols (int): Nombre de colonnes de la grille.
        recompense_mouvement (float): Récompense pour chaque pas non terminal.
        etats_terminaux (dict): Dictionnaire mappant les coordonnées (l,c) à leur récompense terminale.
        obstacles (list): Liste de coordonnées (l,c) des obstacles (murs).
        prob_intention (float): Probabilité que l'agent aille dans la direction choisie.

    Returns:
        dict: Un dictionnaire contenant tous les composants de l'environnement:
              P, R, coordonnees, nb_etat, nb_action, mappings d'états, indices terminaux, etc.
    """
    if etats_terminaux is None:
        etats_terminaux = {}
    if obstacles is None:
        obstacles = []

    # 1. Définition des états et des mappings
    coordonnees = []
    for i in range(lignes):
        for j in range(cols):
            if (i, j) not in obstacles:
                coordonnees.append((i, j))

    etat_en_index = {j: i for i, j in enumerate(coordonnees)}
    index_en_etat = {i: j for i, j in enumerate(coordonnees)}
    nb_etat = len(coordonnees)

    # 2. Définition des actions
    action = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # H, B, G, D
    nb_action = len(action)
    action_perp = {0: (2, 3), 1: (3, 2), 2: (1, 0), 3: (0, 1)}

    # 3. Définition des états terminaux
    indices_etats_terminaux = [etat_en_index[coord] for coord in etats_terminaux if coord in etat_en_index]
    valeurs_terminaux = {etat_en_index[coord]: val for coord, val in etats_terminaux.items() if coord in etat_en_index}
    indices_etats_non_terminaux = [i for i in range(nb_etat) if i not in indices_etats_terminaux]

    # 4. Matrice de récompenses R
    # La récompense est associée à la transition (s, a), menant à R(s,a)
    # Dans ce modèle simple, R(s,a) est constant pour tous les a depuis un état s non-terminal.
    R = np.full((nb_etat, nb_action), recompense_mouvement)
    for s_term_idx in indices_etats_terminaux:
        R[s_term_idx, :] = 0.0  # Aucune récompense une fois dans un état terminal

    # 5. Matrice de transition P
    P = np.zeros((nb_etat, nb_action, nb_etat))
    prob_perp = (1 - prob_intention) / 2

    for etat_courant_idx in range(nb_etat):
        # Si l'état est terminal, il est absorbant
        if etat_courant_idx in indices_etats_terminaux:
            P[etat_courant_idx, :, etat_courant_idx] = 1.0
            continue

        etaty_courant, etatx_courant = index_en_etat[etat_courant_idx]

        for action_courant_idx in range(nb_action):
            sorties = [
                (action_courant_idx, prob_intention),
                (action_perp[action_courant_idx][0], prob_perp),
                (action_perp[action_courant_idx][1], prob_perp)
            ]
            for action_idx, prob in sorties:
                if prob == 0: continue

                actiony, actionx = action[action_idx]
                etaty_tente, etatx_tente = etaty_courant + actiony, etatx_courant + actionx

                # Par défaut, on reste sur place (collision)
                etat_resultant_coord = (etaty_courant, etatx_courant)

                # Si le mouvement est valide (dans la grille et pas un obstacle)
                if (0 <= etaty_tente < lignes and
                        0 <= etatx_tente < cols and
                        (etaty_tente, etatx_tente) not in obstacles):
                    etat_resultant_coord = (etaty_tente, etatx_tente)

                etat_resultant_idx = etat_en_index[etat_resultant_coord]
                P[etat_courant_idx, action_courant_idx, etat_resultant_idx] += prob

    return {
        "P": P, "R_sa": R, "recompense_etat": recompense_mouvement,
        "coordonnees": coordonnees, "nb_etat": nb_etat, "nb_action": nb_action,
        "etat_en_index": etat_en_index, "index_en_etat": index_en_etat,
        "indices_terminaux": indices_etats_terminaux,
        "valeurs_terminaux": valeurs_terminaux,
        "indices_non_terminaux": indices_etats_non_terminaux,
        "lignes": lignes, "cols": cols, "gamma": 0.99  # gamma par défaut
    }


def generer_obstacles_M(lignes, cols):
    """
    Génère une liste de coordonnées d'obstacles en forme de 'M'
    pour une grille de taille donnée.
    La forme est conçue pour être reconnaissable sur des grilles de 5x5 ou plus.
    """
    if lignes < 5 or cols < 5:
        # Pour les petites grilles, un 'M' n'est pas possible, on met un obstacle central.
        return [(lignes // 2, cols // 2)]

    obstacles = []
    center_l, center_c = lignes // 2, cols // 2

    # Le 'M' sera dessiné autour du centre
    # Pilier gauche du 'M'
    obstacles.append((center_l - 1, center_c - 2))
    obstacles.append((center_l, center_c - 2))

    # 'V' au centre
    obstacles.append((center_l - 1, center_c - 1))
    obstacles.append((center_l - 1, center_c + 1))
    obstacles.append((center_l, center_c))

    # Pilier droit du 'M'
    obstacles.append((center_l - 1, center_c + 2))
    obstacles.append((center_l, center_c + 2))

    obstacles_valides = [(l, c) for l, c in obstacles if 0 <= l < lignes and 0 <= c < cols]

    return list(set(obstacles_valides))  # U pour éviter les doublons


import time
import numpy as np
import matplotlib.pyplot as plt
import random


# ==============================================================================
# SECTION 1 : Définition de toutes les fonctions nécessaires
# (Agents, création d'environnement, etc.)
# ==============================================================================

def generer_obstacles_M(lignes, cols):
    if lignes < 5 or cols < 5: return []
    obstacles = [];
    cy, cx = lignes // 2, cols // 2
    for r_off, c_off in [(-1, -2), (0, -2), (-1, -1), (-1, 1), (0, 0), (-1, 2), (0, 2)]:
        if 0 <= cy + r_off < lignes and 0 <= cx + c_off < cols:
            obstacles.append((cy + r_off, cx + c_off))
    return list(set(obstacles))


def creer_environnement_grille(
        lignes: int, cols: int, recompense_mouvement: float = -0.04,
        etats_terminaux: dict = None, obstacles: list = None, prob_intention: float = 0.8):
    if etats_terminaux is None: etats_terminaux = {}
    if obstacles is None: obstacles = []
    coordonnees = [(i, j) for i in range(lignes) for j in range(cols) if (i, j) not in obstacles]
    etat_en_index = {j: i for i, j in enumerate(coordonnees)};
    index_en_etat = {i: j for i, j in enumerate(coordonnees)}
    nb_etat = len(coordonnees);
    action = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)};
    nb_action = len(action)
    action_perp = {0: (2, 3), 1: (3, 2), 2: (1, 0), 3: (0, 1)}
    indices_terminaux = [etat_en_index[coord] for coord in etats_terminaux if coord in etat_en_index]
    valeurs_terminaux = {etat_en_index[coord]: val for coord, val in etats_terminaux.items() if coord in etat_en_index}
    indices_non_terminaux = [i for i in range(nb_etat) if i not in indices_terminaux]
    R_sa = np.full((nb_etat, nb_action), recompense_mouvement)
    for s_term_idx in indices_terminaux: R_sa[s_term_idx, :] = 0.0
    P = np.zeros((nb_etat, nb_action, nb_etat));
    prob_perp = (1 - prob_intention) / 2
    for s_idx in range(nb_etat):
        if s_idx in indices_terminaux: P[s_idx, :, s_idx] = 1.0; continue
        y, x = index_en_etat[s_idx]
        for a_idx in range(nb_action):
            sorties = [(a_idx, prob_intention), (action_perp[a_idx][0], prob_perp), (action_perp[a_idx][1], prob_perp)]
            for act_res, prob in sorties:
                dy, dx = action[act_res];
                ny, nx = y + dy, x + dx
                res_coord = (y, x)
                if 0 <= ny < lignes and 0 <= nx < cols and (ny, nx) not in obstacles: res_coord = (ny, nx)
                res_idx = etat_en_index[res_coord]
                P[s_idx, a_idx, res_idx] += prob
    return {"P": P, "R_sa": R_sa, "recompense_etat": recompense_mouvement, "nb_etat": nb_etat, "nb_action": nb_action,
            "etat_en_index": etat_en_index, "index_en_etat": index_en_etat, "indices_terminaux": indices_terminaux,
            "valeurs_terminaux": valeurs_terminaux, "indices_non_terminaux": indices_non_terminaux, "gamma": 1.0}


def valeur_optimal_iteration(P, gamma, R_etat, indices_nt, idx_plus_1, idx_moins_1, val_etat_plus_1, val_etat_moins_1,
                             max_iter):
    V, _ = valeur_optimal_iteration_base(P, gamma, R_etat, indices_nt,
                                         {idx_plus_1: val_etat_plus_1, idx_moins_1: val_etat_moins_1},
                                         max_iter=max_iter)
    _, pi_dict = valeur_optimal_iteration_base(P, gamma, R_etat, indices_nt,
                                               {idx_plus_1: val_etat_plus_1, idx_moins_1: val_etat_moins_1}, V_init=V)
    return V, pi_dict


def valeur_optimal_iteration_base(P, gamma, R, indices_non_terminaux, valeurs_terminaux, epsilon=1e-5, max_iter=10000,
                                  V_init=None):
    nb_etats, nb_actions = P.shape[0], P.shape[1]
    V = np.zeros(nb_etats) if V_init is None else V_init.copy()
    for idx, val in valeurs_terminaux.items(): V[idx] = val
    for i in range(max_iter):
        V_old = V.copy()
        Q_s = np.full((nb_etats, nb_actions), R) + gamma * (P @ V)
        V_new = np.max(Q_s, axis=1)
        for idx, val in valeurs_terminaux.items(): V_new[idx] = val
        if np.max(np.abs(V_new - V_old)) < epsilon: break
        V = V_new
    Q_final = np.full((nb_etats, nb_actions), R) + gamma * (P @ V)
    pi_array = np.argmax(Q_final, axis=1)
    pi_dict = {i: pi_array[idx] for i, idx in enumerate(indices_non_terminaux)}
    return V, pi_dict


def valeur_politique_livre(pi_non_terminal, R_sa, P, gamma, indices_nt, idx_plus_1, val_plus_1, idx_moins_1,
                           val_moins_1):
    nb_nt = len(indices_nt)
    P_nt = np.zeros((nb_nt, nb_nt))
    R_prime = np.zeros(nb_nt)
    for i, s_idx in enumerate(indices_nt):
        a = pi_non_terminal[i]
        R_prime[i] = R_sa[s_idx, a]
        R_prime[i] += gamma * P[s_idx, a, idx_plus_1] * val_plus_1
        R_prime[i] += gamma * P[s_idx, a, idx_moins_1] * val_moins_1
        for j, s_prime_idx in enumerate(indices_nt):
            P_nt[i, j] = P[s_idx, a, s_prime_idx]
    V_nt = np.linalg.solve(np.eye(nb_nt) - gamma * P_nt, R_prime)
    return V_nt

def agent_passif_terminal(V0, pi_nt, fonction_pas_suivant, nb_etats, indices_nt, indices_t, etat_ini_idx, gamma,
                          N_max_iter, **kwargs):
    politique_env = np.zeros(nb_etats, dtype=int)
    for i, idx in enumerate(indices_nt): politique_env[idx] = pi_nt[i]
    s = etat_ini_idx
    for _ in range(N_max_iter):
        s, _ = fonction_pas_suivant(s, politique_env)
        if s in indices_t: s = etat_ini_idx
    return None, None


def agent_adp_passif(pi_nt, fonction_pas_suivant, gamma, nb_etats, indices_nt, indices_t, idx_plus_1, idx_moins_1,
                     val_plus_1, val_moins_1, etat_ini_idx, nb_episodes_apprentissage):
    politique_env = np.zeros(nb_etats, dtype=int)
    for i, idx in enumerate(indices_nt): politique_env[idx] = pi_nt[i]
    s = etat_ini_idx
    for _ in range(nb_episodes_apprentissage):
        for _ in range(100):  # Simuler un épisode
            s, _ = fonction_pas_suivant(s, politique_env)
            if s in indices_t: s = etat_ini_idx; break
    return None


def agent_actif_Q(fonction_pas_suivant, nb_etats, nb_actions, indices_t, indices_nt, gamma, nb_episodes, epsilon=0.2):
    for _ in range(nb_episodes):
        s = random.choice(indices_nt)
        for _ in range(100):
            if s in indices_t: break
            a = random.randint(0, nb_actions - 1) if random.random() < epsilon else 0
            s, _ = fonction_pas_suivant(s, a)
    return None


# ==============================================================================
# SECTION 2 : SCRIPT PRINCIPAL DE L'EXPÉRIENCE
# ==============================================================================
if __name__ == "__main__":
    SIZES_TO_TEST = [(3, 4), (5, 5), (7, 7), (10, 10)]
    CONFIGURATIONS = [
        {"name": "Sans Obstacles (R=-0.04)", "obstacles": lambda l, c: [], "reward": -0.04},
        {"name": "Avec Obstacles 'M' (R=-0.04)", "obstacles": generer_obstacles_M, "reward": -0.04},
        {"name": "Avec Obstacles 'M' (R=0.0)", "obstacles": generer_obstacles_M, "reward": 0.0},
    ]
    ALGOS_TO_TEST = ["Itération Valeur", "Évaluation Politique", "TD Passif", "ADP Passif", "Q-Learning"]

    all_results = {config['name']: {algo: [] for algo in ALGOS_TO_TEST} for config in CONFIGURATIONS}
    all_nb_etats = {config['name']: [] for config in CONFIGURATIONS}

    for config in CONFIGURATIONS:
        print(f"\n\n===== DÉBUT CONFIGURATION : {config['name']} =====")
        for lignes, cols in SIZES_TO_TEST:
            print(f"\n--- Grille {lignes}x{cols} ---")

            # --- Création de l'environnement ---
            etats_terminaux_config = {(0, cols - 1): 1.0}
            if lignes > 1: etats_terminaux_config[(1, cols - 1)] = -1.0

            env_params = {
                "lignes": lignes, "cols": cols, "recompense_mouvement": config['reward'],
                "etats_terminaux": etats_terminaux_config, "obstacles": config['obstacles'](lignes, cols)
            }
            env = creer_environnement_grille(**env_params)

            # Définir l'état de départ
            etat_initial_coord = (lignes - 1, 0)
            if etat_initial_coord in config['obstacles'](lignes, cols) or etat_initial_coord not in env[
                'etat_en_index']:
                etat_initial_coord = (lignes - 1, 1) if (lignes - 1, 1) in env['etat_en_index'] else \
                    list(env['etat_en_index'].keys())[0]

            all_nb_etats[config['name']].append(env['nb_etat'])

            # --- Extraction des variables de l'environnement ---
            P_env, R_sa_env, R_etat_env, nb_etat_env, nb_action_env = env["P"], env["R_sa"], env["recompense_etat"], \
                env["nb_etat"], env["nb_action"]
            indices_nt, indices_t, val_t, etat_en_index_env = env["indices_non_terminaux"], env["indices_terminaux"], \
                env["valeurs_terminaux"], env["etat_en_index"]

            # Gérer les cas où il n'y a pas deux terminaux
            idx_plus_1 = next((k for k, v in val_t.items() if v > 0), indices_t[0] if indices_t else 0)
            idx_moins_1 = next((k for k, v in val_t.items() if v < 0), idx_plus_1)
            val_plus_1 = val_t.get(idx_plus_1, 0)
            val_moins_1 = val_t.get(idx_moins_1, 0)
            etat_ini_idx = etat_en_index_env[etat_initial_coord]

            # --- Lancement des algorithmes ---

            # 1. Itération de la Valeur (VI)
            start_time = time.time()
            V_opti_vi, pi_opti_vi_dict = valeur_optimal_iteration(P_env, 1.0, R_etat_env, indices_nt, idx_plus_1,
                                                                  idx_moins_1, val_plus_1, val_moins_1, max_iter=2000)
            pi_opti_vi_list = [pi_opti_vi_dict.get(i, 0) for i in range(len(indices_nt))]
            all_results[config['name']]["Itération Valeur"].append(time.time() - start_time)

            # 2. Évaluation Politique
            start_time = time.time()
            try:
                valeur_politique_livre(pi_opti_vi_list, R_sa_env, P_env, 1.0, indices_nt, idx_plus_1, val_plus_1,
                                       idx_moins_1, val_moins_1)
                all_results[config['name']]["Évaluation Politique"].append(time.time() - start_time)
            except np.linalg.LinAlgError:
                all_results[config['name']]["Évaluation Politique"].append(float('inf'))


            # Fonctions de transition pour les agents
            def creer_pas_suivant_local(P, R):
                def passer_etat_suivant(etat, pol):
                    a = pol[etat]
                    s_prime = np.random.choice(len(P), p=P[etat, a, :])
                    r = R[etat, a]
                    return s_prime, r

                return passer_etat_suivant


            passer_etat_suivant_local = creer_pas_suivant_local(P_env, R_sa_env)


            def creer_pas_suivant_action_local(P, R, val_t):
                def passer_etat_suivant_action(etat, act):
                    s_prime = np.random.choice(len(P), p=P[etat, act, :])
                    r = R[etat, act]
                    if s_prime in val_t: r = val_t[s_prime]
                    return s_prime, r

                return passer_etat_suivant_action


            passer_etat_suivant_action_local = creer_pas_suivant_action_local(P_env, R_sa_env, val_t)

            # Budgets fixes
            V0, n_iter_td, n_episodes_q, n_episodes_adp = np.zeros(
                nb_etat_env), 2000 * nb_etat_env, 100 * nb_etat_env, 5 * nb_etat_env

            # 3. TD Passif
            start_time = time.time();
            agent_passif_terminal(V0, pi_opti_vi_list, passer_etat_suivant_local, nb_etat_env, indices_nt, indices_t,
                                  etat_ini_idx, 1.0, n_iter_td);
            all_results[config['name']]["TD Passif"].append(time.time() - start_time)

            # 4. ADP Passif
            start_time = time.time();
            agent_adp_passif(pi_opti_vi_list, passer_etat_suivant_local, 1.0, nb_etat_env, indices_nt, indices_t,
                             idx_plus_1, idx_moins_1, val_plus_1, val_moins_1, etat_ini_idx, n_episodes_adp);
            all_results[config['name']]["ADP Passif"].append(time.time() - start_time)

            # 5. Q-Learning
            start_time = time.time();
            agent_actif_Q(passer_etat_suivant_action_local, nb_etat_env, nb_action_env, indices_t, indices_nt, 1.0,
                          n_episodes_q);
            all_results[config['name']]["Q-Learning"].append(time.time() - start_time)

            print(
                f"  VI:{all_results[config['name']]['Itération Valeur'][-1]:.3f}s | Éval:{all_results[config['name']]['Évaluation Politique'][-1]:.3f}s | TD:{all_results[config['name']]['TD Passif'][-1]:.3f}s | ADP:{all_results[config['name']]['ADP Passif'][-1]:.3f}s | QL:{all_results[config['name']]['Q-Learning'][-1]:.3f}s")

    # ==============================================================================
    # SECTION 3 : AFFICHAGE DES GRAPHIQUES
    # ==============================================================================

    for config in CONFIGURATIONS:

        plt.figure(figsize=(10, 7))

        config_name = config['name']
        results = all_results[config_name]
        nb_etats = all_nb_etats[config_name]


        for algo, temps in results.items():
            etats_valides = [nb_etats[i] for i, t in enumerate(temps) if t != float('inf')]
            temps_valides = [t for t in temps if t != float('inf')]
            if temps_valides:
                plt.plot(etats_valides, temps_valides, marker='o', linestyle='-', label=algo)


        plt.title(f"Comparaison du Temps de Calcul: {config_name}")
        plt.xlabel("Nombre d'États")
        plt.ylabel("Temps de Calcul (secondes)")
        plt.grid(True, which="both", ls="--")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()


    plt.show()








