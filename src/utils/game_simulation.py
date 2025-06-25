import pygame
import numpy as np
import sys
import time
import random

# ############################################################################
# --- CONSTANTES DE CONFIGURATION PYGAME ET JEU ---
# ############################################################################

LOGICAL_WIDTH = 1200
LOGICAL_HEIGHT = 800
BASE_CELL_SIZE = 80
SCORE_TABLE_WIDTH = 250
MINI_GRID_CELL_SIZE = 25

pygame.init()

screen = pygame.display.set_mode((LOGICAL_WIDTH, LOGICAL_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("GridWorld RL (F11 pour Plein Écran)")
game_surface = pygame.Surface((LOGICAL_WIDTH, LOGICAL_HEIGHT))
is_fullscreen = False

FONT_NAME = "Arial"
try:
    font_menu_titre = pygame.font.SysFont(FONT_NAME, 48);
    font_info = pygame.font.SysFont(FONT_NAME, 26);
    font_score_table = pygame.font.SysFont(FONT_NAME, 18);
    font_slider_label = pygame.font.SysFont(FONT_NAME, 18);
    font_petite = pygame.font.SysFont(FONT_NAME, 16);
    font_editor = pygame.font.SysFont(FONT_NAME, 22);
    font_menu_option = pygame.font.SysFont(FONT_NAME, 28);
    font_menu_bouton = pygame.font.SysFont(FONT_NAME, 32)
except pygame.error:
    print(f"Warning: Font '{FONT_NAME}' not found. Falling back to default fonts.");
    font_menu_titre = pygame.font.Font(None, 52);
    font_info = pygame.font.Font(None, 28);
    font_score_table = pygame.font.Font(None, 22);
    font_slider_label = pygame.font.Font(None, 20);
    font_petite = pygame.font.Font(None, 18);
    font_editor = pygame.font.Font(None, 24);
    font_menu_option = pygame.font.Font(None, 30);
    font_menu_bouton = pygame.font.Font(None, 34)

COLOR_WHITE = (255, 255, 255);
COLOR_BLACK = (0, 0, 0);
COLOR_GREY_DARK = (100, 100, 100);
COLOR_RED = (255, 0, 0);
COLOR_GREEN = (0, 150, 0);
COLOR_BLUE = (0, 0, 255);
COLOR_PURPLE = (100, 0, 100);
COLOR_ORANGE = (255, 165, 0);
START_COLOR_BG_JEU = (200, 255, 200);
MENU_BG_COLOR = (50, 50, 80);
SCORE_TABLE_BG_COLOR = (25, 40, 80);
SCORE_TABLE_TEXT_COLOR = (240, 240, 240)
try:
    agent_sprite_original = pygame.image.load("rogue.png").convert_alpha();
    agent_sprite_hurt = pygame.image.load("hurt2.png").convert_alpha()
except pygame.error as e:
    print(f"Warning: Could not load sprite images ({e}). Agent will be a red circle.");
    agent_sprite_original = None;
    agent_sprite_hurt = None

# --- VARIABLES GLOBALES ET DE CONFIGURATION ---
LIGNES_GRILLE, COLS_GRILLE = 3, 4;
coordonnees = [];
etat_en_index = {};
index_en_etat = {};
nb_etat = 0;
indices_etats_terminaux = {};
indices_etats_non_terminaux = [];
nb_etats_non_term = len(indices_etats_non_terminaux)
action_deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)};
action_idx_en_fleche = {0: "↑", 1: "↓", 2: "←", 3: "→"};
nb_action = len(action_deltas);
P_ENV = np.array([]);
R_ENV = np.array([]);
config_etat_initial_idx_agent = -1;
obstacle_coords = [];
config_nom_politique = "Optimale";
politiques_disponibles = ["Optimale", "Aléatoire", "Personnalisée", "Haut"];
config_recompense_non_terminal = -0.04;
config_mode_jeu_initial = "Libre";
modes_jeu_disponibles = ["Libre", "Auto", "AgentPassifTD", "Agent Actif (Q-L)"];
pi_perso_non_term_liste = [];
ETAT_MENU, ETAT_JEU, ETAT_EDITER_POLITIQUE, ETAT_EDITER_GRILLE = 0, 1, 2, 3;
etat_du_jeu_actuel = ETAT_MENU;
clock = pygame.time.Clock();
agent_etat_idx = -1;
score_total_episode_actuel = 0.0;
dernier_message = "Bienvenue ! Configurez et lancez la simulation.";
partie_terminee_message = "";
reset_episode_needed = False;
mode_automatique = False;
temps_derniere_action_auto = 0;
dernier_scores_episodes = [];
afficher_politique, afficher_valeurs_etat = True, True;
mode_agent_passif_td_actif = False;
valeurs_etat_a_afficher = np.array([]);
V_optimal_cibles = np.array([]);
pi_optimale_calculee_non_term = [];
pi_politique_visualisee_et_auto = np.array([]);
nb_passages_etat_td = {};
menu_cliquables_rects = {};
rect_bouton_lancer_sim_menu = pygame.Rect(0, 0, 0, 0);
etat_selectionne_pour_edition_pol = None;
politique_en_edition_non_term = [];
AUTO_MODE_DELAY_JEU = 0.3;
MIN_DELAY, MAX_DELAY = 0.05, 1.0;
slider_rect = None;
handle_rect = None;
slider_is_dragging = False;
Q_table = np.array([]);
Nsa_table = np.array([]);
q_learning_s = None;
q_learning_a = None;
epsilon_q_learning = 0.3;
temp_lignes, temp_cols = LIGNES_GRILLE, COLS_GRILLE;
temp_lignes_str, temp_cols_str = str(LIGNES_GRILLE), str(COLS_GRILLE);
temp_grid_config = {};
editor_input_rects = {};
active_editor_input = None;
shake_timer = 0;
shake_intensity = 5;
is_agent_hurt = False;
hurt_timer = 0;
HURT_DURATION = 0.2
config_gamma = 1.0;
config_gamma_valide = 1.0;
texte_saisie_gamma = "1.00"
config_transition_probas_valide = {0: {0: 0.8, 1: 0.0, 2: 0.1, 3: 0.1}}
textes_saisie_probas = {direction: f"{prob:.1f}" for direction, prob in config_transition_probas_valide[0].items()}
saisie_active = None
proba_feedback_msg = "";
proba_feedback_timer = 0
current_episode_step = 0


# --- ALGORITHMES RL & GESTION ENVIRONNEMENT ---
def valeur_politique_livre(pi_non_terminal, R_mat_sa, P_mat, gamma_val, indices_non_term, term_states_dict):
    nb_etats_non_terminaux = len(indices_non_term);
    P_pi_nt = np.zeros((nb_etats_non_term, nb_etats_non_term));
    R_pi_nt_mod = np.zeros(nb_etats_non_term)
    for i, s_idx in enumerate(indices_non_term):
        action = pi_non_terminal[i];
        R_pi_nt_mod[i] = R_mat_sa[s_idx, action]
        for term_idx, term_val in term_states_dict.items():
            if P_mat[s_idx, action, term_idx] > 1e-9: R_pi_nt_mod[i] += gamma_val * P_mat[
                s_idx, action, term_idx] * term_val
        for j, s_prime_nt in enumerate(indices_non_term): P_pi_nt[i, j] = P_mat[s_idx, action, s_prime_nt]
    try:
        V_nt = np.linalg.solve(np.eye(nb_etats_non_term) - gamma_val * P_pi_nt, R_pi_nt_mod)
    except np.linalg.LinAlgError:
        V_nt = np.linalg.lstsq(np.eye(nb_etats_non_term) - gamma_val * P_pi_nt, R_pi_nt_mod, rcond=None)[0]
    V_complet = np.zeros(nb_etat)
    for term_idx, term_val in term_states_dict.items(): V_complet[term_idx] = term_val
    for i, s_idx in enumerate(indices_non_term): V_complet[s_idx] = V_nt[i]
    return V_complet


def valeur_optimal_iteration(P_mat, gamma, R_cout, indices_non_term, term_states_dict, epsilon=1e-5, max_iter=200):
    V_k = np.zeros(P_mat.shape[0])
    for term_idx, term_val in term_states_dict.items(): V_k[term_idx] = term_val
    pi_opt_dict = {i: 0 for i in range(len(indices_non_term))};
    map_global_to_local = {s: i for i, s in enumerate(indices_non_term)}
    for _ in range(max_iter):
        V_k_plus_1 = V_k.copy();
        delta = 0
        for s in indices_non_term:
            q_valeurs = [R_cout + gamma * np.sum(P_mat[s, a, :] * V_k) for a in range(nb_action)];
            max_q = np.max(q_valeurs);
            delta = max(delta, abs(max_q - V_k[s]));
            V_k_plus_1[s] = max_q;
            pi_opt_dict[map_global_to_local[s]] = int(np.argmax(q_valeurs))
        V_k = V_k_plus_1
        if delta < epsilon: break
    return V_k, pi_opt_dict


def reinitialiser_environnement_complet(lignes, cols, start_pos, obstacles_list, terminals_dict_pos):
    global LIGNES_GRILLE, COLS_GRILLE, coordonnees, etat_en_index, index_en_etat, nb_etat, obstacle_coords, config_etat_initial_idx_agent, indices_etats_terminaux, indices_etats_non_terminaux, P_ENV, R_ENV, Q_table, Nsa_table, valeurs_etat_a_afficher, pi_politique_visualisee_et_auto
    LIGNES_GRILLE, COLS_GRILLE = lignes, cols;
    obstacle_coords = list(obstacles_list);
    coordonnees = [(r, c) for r in range(lignes) for c in range(cols) if (r, c) not in obstacle_coords];
    etat_en_index = {c: i for i, c in enumerate(coordonnees)};
    index_en_etat = {i: c for i, c in enumerate(coordonnees)};
    nb_etat = len(coordonnees);
    config_etat_initial_idx_agent = etat_en_index.get(start_pos, -1)
    if config_etat_initial_idx_agent == -1 and coordonnees: config_etat_initial_idx_agent = 0; print(
        f"Warning: Invalid start position, reset to {index_en_etat.get(0)}")
    indices_etats_terminaux = {etat_en_index[pos]: val for pos, val in terminals_dict_pos.items() if
                               pos in etat_en_index};
    indices_etats_non_terminaux = sorted(list(set(range(nb_etat)) - set(indices_etats_terminaux.keys())));
    P_ENV = np.zeros((nb_etat, nb_action, nb_etat));
    R_ENV = np.zeros((nb_etat, nb_action));
    Q_table = np.zeros((nb_etat, nb_action));
    Nsa_table = np.zeros((nb_etat, nb_action));
    valeurs_etat_a_afficher = np.zeros(nb_etat);
    pi_politique_visualisee_et_auto = np.zeros(nb_etat, dtype=int);
    print("New grid environment initialized.")


def valider_et_appliquer_gamma():
    global config_gamma, config_gamma_valide, texte_saisie_gamma
    try:
        if texte_saisie_gamma and texte_saisie_gamma[-1] != '.':
            val = float(texte_saisie_gamma)
            if 0.0 <= val <= 1.0: config_gamma_valide = val
    except (ValueError, IndexError):
        pass
    config_gamma = config_gamma_valide;
    texte_saisie_gamma = f"{config_gamma:.2f}"


def valider_et_appliquer_probas_transition(silent=False):
    global config_transition_probas_valide, textes_saisie_probas, proba_feedback_msg, proba_feedback_timer
    try:
        new_probas = {direction: float(text) for direction, text in textes_saisie_probas.items()}
        if abs(sum(new_probas.values()) - 1.0) < 1e-5 and all(0.0 <= p <= 1.0 for p in new_probas.values()):
            config_transition_probas_valide[0] = new_probas.copy()
            if not silent: proba_feedback_msg = "Probas appliquées !"; proba_feedback_timer = 2
            return True
        else:
            if not silent: proba_feedback_msg = "Erreur: Somme != 1 ou val hors de [0,1]"; proba_feedback_timer = 2
            return False
    except (ValueError, IndexError):
        if not silent: proba_feedback_msg = "Erreur: Saisie invalide"; proba_feedback_timer = 2
        return False


def reconstruire_P_R_et_politiques():
    global pi_optimale_calculee_non_term, pi_perso_non_term_liste
    valider_et_appliquer_gamma();
    if not valider_et_appliquer_probas_transition(silent=True):
        for dir_idx, prob in config_transition_probas_valide[0].items():
            textes_saisie_probas[dir_idx] = f"{prob:.2f}".rstrip('0').rstrip('.')
    if nb_etat == 0: return
    P_ENV.fill(0)
    base_probas = config_transition_probas_valide[0]
    for s_idx in range(nb_etat):
        if s_idx in indices_etats_terminaux: P_ENV[s_idx, :, s_idx] = 1.0; continue
        r, c = index_en_etat[s_idx]
        for a_idx_intended in range(nb_action):
            for a_idx_deviation_base, prob in base_probas.items():
                if prob <= 0: continue
                effective_outcome = 0
                if a_idx_intended == 0:  # Intention HAUT
                    effective_outcome = a_idx_deviation_base
                elif a_idx_intended == 1:  # Intention BAS
                    if a_idx_deviation_base == 0:
                        effective_outcome = 1
                    elif a_idx_deviation_base == 1:
                        effective_outcome = 0
                    elif a_idx_deviation_base == 2:
                        effective_outcome = 3
                    elif a_idx_deviation_base == 3:
                        effective_outcome = 2
                elif a_idx_intended == 2:  # Intention GAUCHE
                    if a_idx_deviation_base == 0:
                        effective_outcome = 2
                    elif a_idx_deviation_base == 1:
                        effective_outcome = 3
                    elif a_idx_deviation_base == 2:
                        effective_outcome = 1
                    elif a_idx_deviation_base == 3:
                        effective_outcome = 0
                elif a_idx_intended == 3:  # Intention DROITE
                    if a_idx_deviation_base == 0:
                        effective_outcome = 3
                    elif a_idx_deviation_base == 1:
                        effective_outcome = 2
                    elif a_idx_deviation_base == 2:
                        effective_outcome = 0
                    elif a_idx_deviation_base == 3:
                        effective_outcome = 1
                dr, dc = action_deltas[effective_outcome]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < LIGNES_GRILLE and 0 <= nc < COLS_GRILLE and (
                nr, nc) not in obstacle_coords): nr, nc = r, c
                P_ENV[s_idx, a_idx_intended, etat_en_index[(nr, nc)]] += prob
    R_ENV.fill(config_recompense_non_terminal);
    R_ENV[list(indices_etats_terminaux.keys()), :] = 0.0
    V_optimal_cibles, pi_opt_dict = valeur_optimal_iteration(P_ENV, config_gamma, config_recompense_non_terminal,
                                                             indices_etats_non_terminaux, indices_etats_terminaux,
                                                             epsilon=1e-6)
    if pi_opt_dict: pi_optimale_calculee_non_term = [pi_opt_dict.get(i, 0) for i in
                                                     range(len(indices_etats_non_terminaux))]
    if not pi_perso_non_term_liste or len(pi_perso_non_term_liste) != len(
        indices_etats_non_terminaux): pi_perso_non_term_liste = list(pi_optimale_calculee_non_term)


def construire_politique_jeu():
    global pi_politique_visualisee_et_auto
    if nb_etat == 0 or not pi_optimale_calculee_non_term: return
    pi_politique_visualisee_et_auto = np.zeros(nb_etat, dtype=int)
    pi_map = {"Optimale": pi_optimale_calculee_non_term, "Haut": [0] * len(indices_etats_non_terminaux),
              "Aléatoire": [np.random.randint(0, nb_action) for _ in indices_etats_non_terminaux],
              "Personnalisée": pi_perso_non_term_liste if len(pi_perso_non_term_liste) == len(
                  indices_etats_non_terminaux) else pi_optimale_calculee_non_term}
    pi_a_utiliser = pi_map.get(config_nom_politique, pi_optimale_calculee_non_term)
    if len(pi_a_utiliser) == len(indices_etats_non_terminaux):
        for i, s in enumerate(indices_etats_non_terminaux): pi_politique_visualisee_et_auto[s] = pi_a_utiliser[i]


def passer_etat_suivant_env_pygame(s, a):
    if s >= nb_etat or a >= nb_action: return s, 0
    return np.random.choice(nb_etat, p=P_ENV[s, a]), R_ENV[s, a]


reinitialiser_environnement_complet(3, 4, (2, 0), [(1, 1)], {(0, 3): 1.0, (1, 3): -1.0})


# --- FONCTIONS D'AFFICHAGE ---
def get_dynamic_cell_size(rows, cols, available_width, available_height):
    if rows == 0 or cols == 0: return BASE_CELL_SIZE
    cell_w = available_width / cols;
    cell_h = available_height / rows
    return int(min(cell_w, cell_h, BASE_CELL_SIZE))


def dessiner_menu():
    global menu_cliquables_rects, rect_bouton_lancer_sim_menu, proba_feedback_timer
    game_surface.fill(MENU_BG_COLOR);
    w, h = LOGICAL_WIDTH, LOGICAL_HEIGHT
    titre_surf = font_menu_titre.render("Bienvenue au GRIDWORLD RL", True, COLOR_WHITE);
    titre_rect = titre_surf.get_rect(centerx=w / 2, top=40);
    game_surface.blit(titre_surf, titre_rect)
    rect_bouton_lancer_sim_menu = pygame.Rect(0, 0, 280, 50);
    rect_bouton_lancer_sim_menu.center = (w / 2, h - 80);
    pygame.draw.rect(game_surface, COLOR_GREEN, rect_bouton_lancer_sim_menu, 0, 10);
    lancer_surf = font_menu_bouton.render("Lancer Simulation", True, COLOR_WHITE);
    game_surface.blit(lancer_surf, lancer_surf.get_rect(center=rect_bouton_lancer_sim_menu.center))
    content_top = titre_rect.bottom + 50;
    content_bottom = rect_bouton_lancer_sim_menu.top - 50
    menu_cliquables_rects.clear();
    col1_x = 100
    label_start_surf = font_menu_option.render("État initial (clic):", True, (200, 200, 250));
    game_surface.blit(label_start_surf, (col1_x, content_top))
    mini_grid_y = content_top + 40
    for r in range(LIGNES_GRILLE):
        for c in range(COLS_GRILLE):
            pos = (r, c);
            rect = pygame.Rect(col1_x + c * MINI_GRID_CELL_SIZE, mini_grid_y + r * MINI_GRID_CELL_SIZE,
                               MINI_GRID_CELL_SIZE, MINI_GRID_CELL_SIZE);
            color = (230, 230, 230)
            if pos in obstacle_coords:
                color = COLOR_GREY_DARK
            elif etat_en_index.get(pos) == config_etat_initial_idx_agent:
                color = START_COLOR_BG_JEU
            elif etat_en_index.get(pos) in indices_etats_terminaux:
                color = COLOR_ORANGE if indices_etats_terminaux[etat_en_index[pos]] < 0 else COLOR_RED
            pygame.draw.rect(game_surface, color, rect);
            pygame.draw.rect(game_surface, COLOR_BLACK, rect, 1);
            menu_cliquables_rects[f"mini_grid_{r}_{c}"] = rect
    proba_editor_y = mini_grid_y + LIGNES_GRILLE * MINI_GRID_CELL_SIZE + 50
    label_proba_surf = font_menu_option.render("Déviation (action voulue: ↑):", True, (200, 200, 250));
    game_surface.blit(label_proba_surf, (col1_x, proba_editor_y))
    box_size = 80;
    input_w, input_h = 60, 30;
    box_rect = pygame.Rect(col1_x + 120, proba_editor_y + 80, box_size, box_size)
    pygame.draw.rect(game_surface, COLOR_WHITE, box_rect, 2);
    game_surface.blit(font_menu_titre.render("↑", True, COLOR_WHITE), (box_rect.centerx - 15, box_rect.top - 5))
    rect_up = pygame.Rect(box_rect.centerx - input_w / 2, box_rect.top - input_h - 10, input_w, input_h);
    pygame.draw.rect(game_surface, COLOR_WHITE, rect_up, 2 if saisie_active == "proba_0" else 1);
    surf = font_menu_bouton.render(textes_saisie_probas[0], True, COLOR_WHITE);
    game_surface.blit(surf, surf.get_rect(center=rect_up.center));
    menu_cliquables_rects["proba_0"] = rect_up
    rect_down = pygame.Rect(box_rect.centerx - input_w / 2, box_rect.bottom + 10, input_w, input_h);
    pygame.draw.rect(game_surface, COLOR_WHITE, rect_down, 2 if saisie_active == "proba_1" else 1);
    surf = font_menu_bouton.render(textes_saisie_probas[1], True, COLOR_WHITE);
    game_surface.blit(surf, surf.get_rect(center=rect_down.center));
    menu_cliquables_rects["proba_1"] = rect_down
    rect_left = pygame.Rect(box_rect.left - input_w - 10, box_rect.centery - input_h / 2, input_w, input_h);
    pygame.draw.rect(game_surface, COLOR_WHITE, rect_left, 2 if saisie_active == "proba_2" else 1);
    surf = font_menu_bouton.render(textes_saisie_probas[2], True, COLOR_WHITE);
    game_surface.blit(surf, surf.get_rect(center=rect_left.center));
    menu_cliquables_rects["proba_2"] = rect_left
    rect_right = pygame.Rect(box_rect.right + 10, box_rect.centery - input_h / 2, input_w, input_h);
    pygame.draw.rect(game_surface, COLOR_WHITE, rect_right, 2 if saisie_active == "proba_3" else 1);
    surf = font_menu_bouton.render(textes_saisie_probas[3], True, COLOR_WHITE);
    game_surface.blit(surf, surf.get_rect(center=rect_right.center));
    menu_cliquables_rects["proba_3"] = rect_right
    confirm_rect = pygame.Rect(0, 0, 150, 35);
    confirm_rect.midtop = (box_rect.centerx, rect_down.bottom + 20)
    pygame.draw.rect(game_surface, COLOR_BLUE, confirm_rect, 0, 5);
    surf = font_menu_bouton.render("Appliquer", True, COLOR_WHITE);
    game_surface.blit(surf, surf.get_rect(center=confirm_rect.center));
    menu_cliquables_rects["confirm_proba"] = confirm_rect
    if proba_feedback_timer > 0:
        feedback_surf = font_petite.render(proba_feedback_msg, True,
                                           COLOR_GREEN if "appliquées" in proba_feedback_msg else COLOR_RED)
        game_surface.blit(feedback_surf, feedback_surf.get_rect(midtop=confirm_rect.midbottom))
        proba_feedback_timer -= clock.get_time() / 1000.0
    col2_x = 700;
    lh_option = 45
    options_col2 = [("Politique:", "politique", config_nom_politique), ("Mode Jeu:", "mode", config_mode_jeu_initial),
                    ("Gamma (γ):", "gamma", texte_saisie_gamma)]
    for i, (label_text, key, val_text) in enumerate(options_col2):
        label_surf = font_menu_option.render(label_text, True, (200, 200, 250));
        game_surface.blit(label_surf, (col2_x, content_top + i * lh_option));
        rect = pygame.Rect(col2_x + 160, content_top + i * lh_option, 220, 32);
        is_active = saisie_active == key;
        pygame.draw.rect(game_surface, (230, 230, 230) if not is_active else (255, 255, 0), rect, 0, 5)
        val_surf = font_menu_bouton.render(val_text, True, COLOR_BLACK);
        game_surface.blit(val_surf, val_surf.get_rect(center=rect.center));
        menu_cliquables_rects[key] = rect
    edit_pol_surf = font_menu_option.render("Éditer Politique Personnalisée", True, (200, 250, 200));
    edit_pol_rect = edit_pol_surf.get_rect(topleft=(col2_x, content_top + 4 * lh_option));
    game_surface.blit(edit_pol_surf, edit_pol_rect);
    menu_cliquables_rects["edit_pol"] = edit_pol_rect
    edit_grid_surf = font_menu_option.render("Éditer la Grille", True, (200, 250, 200));
    edit_grid_rect = edit_grid_surf.get_rect(topleft=(col2_x, content_top + 5 * lh_option));
    game_surface.blit(edit_grid_surf, edit_grid_rect);
    menu_cliquables_rects["edit_grid"] = edit_grid_rect


def get_shaken_coords(base_x, base_y):
    if shake_timer > 0: return base_x + random.randint(-shake_intensity, shake_intensity), base_y + random.randint(
        -shake_intensity, shake_intensity)
    return base_x, base_y


def draw_grid_backgrounds(base_x, base_y, grid_l, grid_c, cell_size):
    for r in range(grid_l):
        for c in range(grid_c):
            screen_x, screen_y = get_shaken_coords(base_x + c * cell_size, base_y + r * cell_size)
            rect = pygame.Rect(screen_x, screen_y, cell_size, cell_size);
            bg_color = COLOR_WHITE
            if grid_l == LIGNES_GRILLE and grid_c == COLS_GRILLE:
                if etat_en_index.get((r, c)) == config_etat_initial_idx_agent: bg_color = START_COLOR_BG_JEU
            pygame.draw.rect(game_surface, bg_color, rect);
            if (r, c) in obstacle_coords: pygame.draw.rect(game_surface, COLOR_GREY_DARK, rect)
            pygame.draw.rect(game_surface, COLOR_BLACK, rect, 1)


def draw_grid_foregrounds(base_x, base_y, grid_l, grid_c, cell_size, draw_policy_info=True):
    try:
        font_grande_dyn = pygame.font.SysFont(FONT_NAME,
                                              int(cell_size * 0.4)); font_valeur_etat_dyn = pygame.font.SysFont(
            FONT_NAME, int(cell_size * 0.22))
    except pygame.error:
        font_grande_dyn = pygame.font.Font(None, int(cell_size * 0.4)); font_valeur_etat_dyn = pygame.font.Font(None,
                                                                                                                int(cell_size * 0.22))
    for r in range(grid_l):
        for c in range(grid_c):
            etat_idx = etat_en_index.get((r, c));
            if etat_idx is not None:
                rect = pygame.Rect(base_x + c * cell_size, base_y + r * cell_size, cell_size, cell_size)
                if etat_idx in indices_etats_terminaux:
                    val = indices_etats_terminaux[etat_idx];
                    text, color = (f"+{val:.1f}", COLOR_GREEN) if val > 0 else (f"{val:.1f}", COLOR_ORANGE)
                    surf = font_grande_dyn.render(text, True, color);
                    game_surface.blit(surf, surf.get_rect(center=rect.center))
                if draw_policy_info:
                    if afficher_politique and etat_idx not in indices_etats_terminaux:
                        arrow = "?";
                        if config_mode_jeu_initial == "Agent Actif (Q-L)":
                            action = np.argmax(
                                Q_table[etat_idx]) if nb_etat > 0 else 0; arrow = action_idx_en_fleche.get(action, "?")
                        elif len(pi_politique_visualisee_et_auto) > etat_idx:
                            arrow = action_idx_en_fleche.get(pi_politique_visualisee_et_auto[etat_idx], "?")
                        surf = font_valeur_etat_dyn.render(arrow, True, COLOR_BLUE);
                        game_surface.blit(surf, surf.get_rect(center=(rect.centerx, rect.centery - cell_size * 0.25)))
                    if afficher_valeurs_etat:
                        val_text = "0.00"
                        if config_mode_jeu_initial == "Agent Actif (Q-L)":
                            val = np.max(Q_table[
                                             etat_idx]) if etat_idx not in indices_etats_terminaux and nb_etat > 0 else indices_etats_terminaux.get(
                                etat_idx, 0); val_text = f"{val:.2f}"
                        else:
                            if etat_idx < len(
                                valeurs_etat_a_afficher): val_text = f"{valeurs_etat_a_afficher[etat_idx]:.2f}"
                        surf = font_valeur_etat_dyn.render(val_text, True, COLOR_PURPLE);
                        game_surface.blit(surf, surf.get_rect(center=(rect.centerx, rect.centery + cell_size * 0.25)))


def draw_agent(base_x, base_y, cell_size):
    if agent_etat_idx >= 0 and agent_etat_idx in index_en_etat:
        r, c = index_en_etat[agent_etat_idx];
        x, y = get_shaken_coords(base_x + c * cell_size, base_y + r * cell_size)
        sprite_to_draw = agent_sprite_original
        if is_agent_hurt and agent_sprite_hurt: sprite_to_draw = agent_sprite_hurt
        if sprite_to_draw:
            sprite_size = int(cell_size * 1.5);
            scaled_sprite = pygame.transform.scale(sprite_to_draw, (sprite_size, sprite_size))
            game_surface.blit(scaled_sprite, scaled_sprite.get_rect(center=(x + cell_size // 2, y + cell_size // 2)))
        else:
            pygame.draw.circle(game_surface, COLOR_RED, (x + cell_size // 2, y + cell_size // 2), cell_size // 3)


def draw_game_view():
    game_surface.fill(COLOR_WHITE)
    cell_size = get_dynamic_cell_size(LIGNES_GRILLE, COLS_GRILLE, LOGICAL_WIDTH - SCORE_TABLE_WIDTH - 120,
                                      LOGICAL_HEIGHT - 120)
    game_area_width = COLS_GRILLE * cell_size;
    game_area_height = LIGNES_GRILLE * cell_size
    total_width = game_area_width + SCORE_TABLE_WIDTH + 120
    base_x = (LOGICAL_WIDTH - total_width) // 2;
    base_y = (LOGICAL_HEIGHT - game_area_height) // 2
    grid_base_x, grid_base_y = base_x + 40, base_y
    draw_grid_backgrounds(grid_base_x, grid_base_y, LIGNES_GRILLE, COLS_GRILLE, cell_size)
    draw_agent(grid_base_x, grid_base_y, cell_size)
    show_policy = config_mode_jeu_initial != "Libre"
    draw_grid_foregrounds(grid_base_x, grid_base_y, LIGNES_GRILLE, COLS_GRILLE, cell_size, draw_policy_info=show_policy)
    top_panel_rect = pygame.Rect(0, 0, LOGICAL_WIDTH, base_y);
    bottom_panel_rect = pygame.Rect(0, base_y + game_area_height, LOGICAL_WIDTH,
                                    LOGICAL_HEIGHT - (base_y + game_area_height))
    msg_surf = font_info.render(dernier_message, True, COLOR_BLACK);
    game_surface.blit(msg_surf, (top_panel_rect.left + 20, top_panel_rect.centery - 10))
    score_msg = f"Score Actuel: {score_total_episode_actuel:.2f}";
    score_surf = font_info.render(score_msg, True, COLOR_BLACK);
    game_surface.blit(score_surf, (top_panel_rect.left + 20, top_panel_rect.centery + 20))
    help_surf = font_petite.render("'M': Menu, Esc: Quitter, F11: Plein Écran", True, COLOR_BLACK);
    game_surface.blit(help_surf, (20, LOGICAL_HEIGHT - 30))
    global slider_rect, handle_rect
    if partie_terminee_message:
        fin_surf = font_info.render(partie_terminee_message, True, COLOR_RED);
        game_surface.blit(fin_surf, fin_surf.get_rect(centerx=LOGICAL_WIDTH / 2, centery=bottom_panel_rect.centery))
    else:
        slider_w, slider_h = 200, 8;
        slider_x = (LOGICAL_WIDTH - slider_w) / 2;
        slider_y = bottom_panel_rect.centery
        slider_rect = pygame.Rect(slider_x, slider_y - slider_h // 2, slider_w, slider_h)
        handle_w, handle_h = 18, 18;
        percent_fast = (MAX_DELAY - AUTO_MODE_DELAY_JEU) / (MAX_DELAY - MIN_DELAY) if (
                                                                                                  MAX_DELAY - MIN_DELAY) != 0 else 0;
        handle_x = slider_x + percent_fast * (slider_w);
        handle_rect = pygame.Rect(handle_x - handle_w // 2, slider_y - handle_h // 2, handle_w, handle_h)
        pygame.draw.rect(game_surface, COLOR_GREY_DARK, slider_rect, 0, 4);
        pygame.draw.rect(game_surface, COLOR_BLUE, handle_rect, 0, 9)
        label_slow = font_slider_label.render("Lent", True, COLOR_BLACK);
        game_surface.blit(label_slow, (slider_x - 40, slider_y - 10));
        label_fast = font_slider_label.render("Rapide", True, COLOR_BLACK);
        game_surface.blit(label_fast, (slider_x + slider_w + 10, slider_y - 10))
    table_x = base_x + game_area_width + 80;
    table_y = base_y;
    table_h = 40 + 10 * (font_score_table.get_height() + 5)
    pygame.draw.rect(game_surface, SCORE_TABLE_BG_COLOR, (table_x, table_y, SCORE_TABLE_WIDTH, table_h), 0, 10)
    title = font_info.render("Derniers Scores:", True, SCORE_TABLE_TEXT_COLOR);
    game_surface.blit(title, (table_x + 15, table_y + 10))
    lh = font_score_table.get_height() + 5
    for i, score in enumerate(dernier_scores_episodes[-10:]):
        text = f"Ep {len(dernier_scores_episodes) - len(dernier_scores_episodes[-10:]) + i + 1}: {score:.2f}";
        surf = font_score_table.render(text, True, SCORE_TABLE_TEXT_COLOR);
        game_surface.blit(surf, (table_x + 20, table_y + 50 + i * lh))


def dessiner_editeur_politique():
    global pi_politique_visualisee_et_auto
    game_surface.fill(COLOR_WHITE);
    nb_etats_non_terminaux = len(indices_etats_non_terminaux)
    cell_size = get_dynamic_cell_size(LIGNES_GRILLE, COLS_GRILLE, LOGICAL_WIDTH - 80, LOGICAL_HEIGHT - 200)
    grid_w, grid_h = COLS_GRILLE * cell_size, LIGNES_GRILLE * cell_size;
    panel_h = 150;
    total_h = grid_h + panel_h
    base_x = (LOGICAL_WIDTH - grid_w) // 2;
    base_y = (LOGICAL_HEIGHT - total_h) // 2
    panel_rect = pygame.Rect(0, base_y, LOGICAL_WIDTH, panel_h)
    msg = "Éditeur: Clic case, puis Flèches. 'S' Sauver & Quitter, 'Q' Quitter."
    if etat_selectionne_pour_edition_pol is not None and etat_selectionne_pour_edition_pol in indices_etats_non_terminaux:
        if indices_etats_non_terminaux.index(etat_selectionne_pour_edition_pol) < len(politique_en_edition_non_term):
            action = politique_en_edition_non_term[indices_etats_non_terminaux.index(etat_selectionne_pour_edition_pol)]
            msg = f"État {index_en_etat[etat_selectionne_pour_edition_pol]} (action: {action_idx_en_fleche.get(action, '?')}). Flèches pour changer."
    surf = font_info.render(msg, True, COLOR_BLACK);
    game_surface.blit(surf, surf.get_rect(center=panel_rect.center))
    grid_base_y = base_y + panel_h;
    backup_pi = np.copy(pi_politique_visualisee_et_auto)
    if nb_etats_non_terminaux > 0:
        for i, s in enumerate(indices_etats_non_terminaux):
            if i < len(politique_en_edition_non_term): pi_politique_visualisee_et_auto[s] = \
            politique_en_edition_non_term[i]
    draw_grid_backgrounds(base_x, grid_base_y, LIGNES_GRILLE, COLS_GRILLE, cell_size)
    draw_grid_foregrounds(base_x, grid_base_y, LIGNES_GRILLE, COLS_GRILLE, cell_size, draw_policy_info=True)
    pi_politique_visualisee_et_auto = backup_pi
    if etat_selectionne_pour_edition_pol is not None:
        r, c = index_en_etat[etat_selectionne_pour_edition_pol];
        x, y = base_x + c * cell_size, grid_base_y + r * cell_size
        pygame.draw.rect(game_surface, COLOR_RED, (x, y, cell_size, cell_size), 3)


def dessiner_editeur_grille():
    global editor_input_rects
    game_surface.fill(MENU_BG_COLOR);
    w, h = LOGICAL_WIDTH, LOGICAL_HEIGHT
    cell_size = get_dynamic_cell_size(temp_lignes, temp_cols, w - 100, h - 250)
    grid_w_temp, grid_h_temp = temp_cols * cell_size, temp_lignes * cell_size
    titre_surf = font_menu_titre.render("Éditeur de Grille", True, COLOR_WHITE);
    titre_rect = titre_surf.get_rect(centerx=w / 2, y=50);
    game_surface.blit(titre_surf, titre_rect)
    instr_surf = font_editor.render("Clic Gauche pour changer type. 'S' pour Sauvegarder, 'Q' pour Quitter.", True,
                                    COLOR_WHITE);
    instr_rect = instr_surf.get_rect(centerx=w / 2, y=titre_rect.bottom + 20);
    game_surface.blit(instr_surf, instr_rect)
    input_y = instr_rect.bottom + 30
    rows_label = font_editor.render("Lignes:", True, COLOR_WHITE);
    game_surface.blit(rows_label, (w / 2 - 150, input_y));
    rows_rect = pygame.Rect(w / 2 - 80, input_y - 5, 60, 30);
    pygame.draw.rect(game_surface, COLOR_WHITE, rows_rect, 2 if active_editor_input == "rows" else 1);
    rows_surf = font_editor.render(temp_lignes_str, True, COLOR_WHITE);
    game_surface.blit(rows_surf, rows_surf.get_rect(center=rows_rect.center));
    editor_input_rects['rows'] = rows_rect
    cols_label = font_editor.render("Colonnes:", True, COLOR_WHITE);
    game_surface.blit(cols_label, (w / 2 + 20, input_y));
    cols_rect = pygame.Rect(w / 2 + 120, input_y - 5, 60, 30);
    pygame.draw.rect(game_surface, COLOR_WHITE, cols_rect, 2 if active_editor_input == "cols" else 1);
    cols_surf = font_editor.render(temp_cols_str, True, COLOR_WHITE);
    game_surface.blit(cols_surf, cols_surf.get_rect(center=cols_rect.center));
    editor_input_rects['cols'] = cols_rect
    grid_base_x = (w - grid_w_temp) // 2;
    grid_base_y = input_y + 60
    try:
        font_grande_dyn = pygame.font.SysFont(FONT_NAME, int(cell_size * 0.4))
    except pygame.error:
        font_grande_dyn = pygame.font.Font(None, int(cell_size * 0.4))
    for r in range(temp_lignes):
        for c in range(temp_cols):
            rect = pygame.Rect(grid_base_x + c * cell_size, grid_base_y + r * cell_size, cell_size, cell_size)
            cell_type = temp_grid_config.get((r, c), 0)
            color_map = {0: COLOR_WHITE, 1: START_COLOR_BG_JEU, 2: COLOR_GREY_DARK, 3: COLOR_RED, 4: COLOR_ORANGE};
            text_map = {3: "+1.0", 4: "-1.0"}
            pygame.draw.rect(game_surface, color_map.get(cell_type, COLOR_WHITE), rect)
            if cell_type in text_map:
                text_surf = font_grande_dyn.render(text_map[cell_type], True, COLOR_WHITE);
                game_surface.blit(text_surf, text_surf.get_rect(center=rect.center))
            pygame.draw.rect(game_surface, COLOR_BLACK, rect, 1);
            editor_input_rects[f"cell_{r}_{c}"] = rect


def setup_jeu():
    global agent_etat_idx, mode_automatique, mode_agent_passif_td_actif, valeurs_etat_a_afficher, nb_passages_etat_td, q_learning_s, q_learning_a, Q_table, Nsa_table
    nb_etats_non_terminaux = len(indices_etats_non_terminaux);
    valeurs_etat_a_afficher = np.zeros(nb_etat);
    construire_politique_jeu()
    if nb_etat > 0: Q_table.fill(0); Nsa_table.fill(0)
    q_learning_s, q_learning_a = None, None;
    nb_passages_etat_td = {i: 0 for i in indices_etats_non_terminaux}
    if config_nom_politique == "Optimale":
        valeurs_etat_a_afficher = np.copy(V_optimal_cibles)
    elif nb_etats_non_terminaux > 0:
        pi_eval = [pi_politique_visualisee_et_auto[idx] for idx in indices_etats_non_terminaux]
        if pi_eval and len(pi_eval) == len(
            indices_etats_non_terminaux): valeurs_etat_a_afficher = valeur_politique_livre(pi_eval, R_ENV, P_ENV,
                                                                                           config_gamma,
                                                                                           indices_etats_non_terminaux,
                                                                                           indices_etats_terminaux)
    mode_automatique = (config_mode_jeu_initial == "Auto");
    mode_agent_passif_td_actif = (config_mode_jeu_initial == "AgentPassifTD")
    if mode_agent_passif_td_actif and nb_etat > 0:
        valeurs_etat_a_afficher = np.zeros(nb_etat)
        for term_idx, term_val in indices_etats_terminaux.items(): valeurs_etat_a_afficher[term_idx] = term_val
    reset_episode_pygame()


def reset_episode_pygame():
    global agent_etat_idx, score_total_episode_actuel, dernier_message, partie_terminee_message, reset_episode_needed, q_learning_s, q_learning_a, current_episode_step
    if reset_episode_needed and score_total_episode_actuel != 0:
        dernier_scores_episodes.append(score_total_episode_actuel)
        if len(dernier_scores_episodes) > 10: dernier_scores_episodes.pop(0)
    agent_etat_idx = config_etat_initial_idx_agent;
    score_total_episode_actuel = 0.0;
    dernier_message = f"Mode: {config_mode_jeu_initial} | Politique: {config_nom_politique}";
    if config_mode_jeu_initial == "Libre": dernier_message += " | Utilisez les flèches."
    partie_terminee_message = "";
    reset_episode_needed = False;
    q_learning_s, q_learning_a = None, None
    current_episode_step = 0


def get_logical_pos(physical_pos):
    screen_w, screen_h = screen.get_size();
    logical_aspect = LOGICAL_WIDTH / LOGICAL_HEIGHT
    if screen_w == 0 or screen_h == 0: return (0, 0)
    screen_aspect = screen_w / screen_h
    if screen_aspect > logical_aspect:
        scale = screen_h / LOGICAL_HEIGHT;
        offset_x = (screen_w - LOGICAL_WIDTH * scale) / 2
        if scale == 0: return (0, 0)
        return ((physical_pos[0] - offset_x) / scale, physical_pos[1] / scale)
    else:
        scale = screen_w / LOGICAL_WIDTH;
        offset_y = (screen_h - LOGICAL_HEIGHT * scale) / 2
        if scale == 0: return (0, 0)
        return (physical_pos[0] / scale, (physical_pos[1] - offset_y) / scale)


def gerer_clic_menu(pos):
    global saisie_active
    logical_pos = get_logical_pos(pos);
    is_input_field_clicked = False
    global config_nom_politique, config_mode_jeu_initial, etat_du_jeu_actuel, config_etat_initial_idx_agent, politique_en_edition_non_term, temp_lignes, temp_cols, temp_lignes_str, temp_cols_str, temp_grid_config
    if rect_bouton_lancer_sim_menu.collidepoint(
        logical_pos): reconstruire_P_R_et_politiques(); setup_jeu(); etat_du_jeu_actuel = ETAT_JEU; return
    for key, rect in list(menu_cliquables_rects.items()):
        if rect.collidepoint(logical_pos):
            saisie_active = key;
            is_input_field_clicked = True
            if key == 'politique':
                config_nom_politique = politiques_disponibles[
                    (politiques_disponibles.index(config_nom_politique) + 1) % len(
                        politiques_disponibles)]; saisie_active = None
            elif key == 'mode':
                config_mode_jeu_initial = modes_jeu_disponibles[
                    (modes_jeu_disponibles.index(config_mode_jeu_initial) + 1) % len(
                        modes_jeu_disponibles)]; saisie_active = None
            elif key == 'edit_pol':
                reconstruire_P_R_et_politiques(); politique_en_edition_non_term = list(pi_perso_non_term_liste) if len(
                    pi_perso_non_term_liste) == len(indices_etats_non_terminaux) else list(
                    pi_optimale_calculee_non_term); etat_du_jeu_actuel = ETAT_EDITER_POLITIQUE; saisie_active = None
            elif key == 'edit_grid':
                temp_lignes, temp_cols = LIGNES_GRILLE, COLS_GRILLE;
                temp_lignes_str, temp_cols_str = str(LIGNES_GRILLE), str(COLS_GRILLE);
                temp_grid_config = {}
                start_pos = index_en_etat.get(config_etat_initial_idx_agent);
                if start_pos: temp_grid_config[start_pos] = 1
                for obs_pos in obstacle_coords: temp_grid_config[obs_pos] = 2
                for term_pos_idx, term_val in indices_etats_terminaux.items():
                    term_pos = index_en_etat.get(term_pos_idx)
                    if term_pos: temp_grid_config[term_pos] = 3 if term_val > 0 else 4
                etat_du_jeu_actuel = ETAT_EDITER_GRILLE;
                saisie_active = None
            elif key.startswith("mini_grid_"):
                *_, r, c = key.split("_");
                r, c = int(r), int(c)
                if (r, c) not in obstacle_coords and etat_en_index.get((r, c),
                                                                       -1) not in indices_etats_terminaux: config_etat_initial_idx_agent = etat_en_index.get(
                    (r, c), config_etat_initial_idx_agent)
                saisie_active = None
            elif key == "confirm_proba":
                valider_et_appliquer_probas_transition(); saisie_active = None
            return
    if not is_input_field_clicked and saisie_active is not None:
        if saisie_active == 'gamma': valider_et_appliquer_gamma()
        saisie_active = None


def gerer_evenements_editeur_politique(event):
    global etat_du_jeu_actuel, etat_selectionne_pour_edition_pol, politique_en_edition_non_term, config_nom_politique, pi_perso_non_term_liste
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        logical_pos = get_logical_pos(event.pos)
        cell_size = get_dynamic_cell_size(LIGNES_GRILLE, COLS_GRILLE, LOGICAL_WIDTH - 80, LOGICAL_HEIGHT - 200)
        grid_w, grid_h = COLS_GRILLE * cell_size, LIGNES_GRILLE * cell_size;
        panel_h = 150;
        total_h = grid_h + panel_h
        base_x = (LOGICAL_WIDTH - grid_w) // 2;
        grid_base_y = ((LOGICAL_HEIGHT - total_h) // 2) + panel_h
        for r in range(LIGNES_GRILLE):
            for c in range(COLS_GRILLE):
                s = etat_en_index.get((r, c))
                if s is not None and s in indices_etats_non_terminaux:
                    rect = pygame.Rect(base_x + c * cell_size, grid_base_y + r * cell_size, cell_size, cell_size)
                    if rect.collidepoint(logical_pos): etat_selectionne_pour_edition_pol = s; return
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_q:
            etat_du_jeu_actuel = ETAT_MENU
        elif event.key == pygame.K_s:
            pi_perso_non_term_liste = list(politique_en_edition_non_term);
            config_nom_politique = "Personnalisée";
            etat_du_jeu_actuel = ETAT_MENU;
            print("Politique personnalisée sauvegardée.")
        if etat_selectionne_pour_edition_pol is not None and etat_selectionne_pour_edition_pol in indices_etats_non_terminaux:
            idx = indices_etats_non_terminaux.index(etat_selectionne_pour_edition_pol)
            action_map = {pygame.K_UP: 0, pygame.K_DOWN: 1, pygame.K_LEFT: 2, pygame.K_RIGHT: 3}
            if event.key in action_map and idx < len(politique_en_edition_non_term): politique_en_edition_non_term[
                idx] = action_map[event.key]


def gerer_evenements_editeur_grille(event):
    global etat_du_jeu_actuel, temp_grid_config, temp_lignes, temp_cols, temp_lignes_str, temp_cols_str, active_editor_input
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_q:
            etat_du_jeu_actuel = ETAT_MENU
        elif event.key == pygame.K_s:
            start_pos, obstacles, terminals = None, [], {}
            for pos, type_id in temp_grid_config.items():
                if type_id == 1:
                    start_pos = pos
                elif type_id == 2:
                    obstacles.append(pos)
                elif type_id == 3:
                    terminals[pos] = 1.0
                elif type_id == 4:
                    terminals[pos] = -1.0
            if start_pos and terminals:
                reinitialiser_environnement_complet(temp_lignes, temp_cols, start_pos, obstacles,
                                                    terminals); etat_du_jeu_actuel = ETAT_MENU
            else:
                print("Erreur: Il faut au moins un point de départ et un état terminal.")
        elif active_editor_input:
            if event.key == pygame.K_BACKSPACE:
                if active_editor_input == 'rows':
                    temp_lignes_str = temp_lignes_str[:-1]
                else:
                    temp_cols_str = temp_cols_str[:-1]
            elif event.unicode.isdigit():
                if active_editor_input == 'rows':
                    temp_lignes_str += event.unicode
                else:
                    temp_cols_str += event.unicode
            elif event.key == pygame.K_RETURN:
                active_editor_input = None
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        logical_pos = get_logical_pos(event.pos)
        if editor_input_rects['rows'].collidepoint(logical_pos):
            active_editor_input = 'rows'
        elif editor_input_rects['cols'].collidepoint(logical_pos):
            active_editor_input = 'cols'
        else:
            active_editor_input = None
            cell_size = get_dynamic_cell_size(temp_lignes, temp_cols, LOGICAL_WIDTH - 100, LOGICAL_HEIGHT - 250)
            grid_w_temp = temp_cols * cell_size;
            grid_base_x = (LOGICAL_WIDTH - grid_w_temp) // 2;
            grid_base_y = 210
            for r in range(temp_lignes):
                for c in range(temp_cols):
                    rect = pygame.Rect(grid_base_x + c * cell_size, grid_base_y + r * cell_size, cell_size, cell_size)
                    if rect.collidepoint(logical_pos):
                        current_type = temp_grid_config.get((r, c), 0);
                        temp_grid_config[(r, c)] = (current_type + 1) % 5
                        if temp_grid_config[(r, c)] == 1:
                            for pos, type_id in list(temp_grid_config.items()):
                                if type_id == 1 and pos != (r, c): temp_grid_config[pos] = 0
                        return
    if not active_editor_input:
        try:
            new_l = int(temp_lignes_str) if temp_lignes_str else temp_lignes;
            new_c = int(temp_cols_str) if temp_cols_str else temp_cols
            temp_lignes = max(1, min(20, new_l));
            temp_cols = max(1, min(20, new_c))
        except ValueError:
            pass
        temp_lignes_str, temp_cols_str = str(temp_lignes), str(temp_cols)


def render_to_screen():
    screen.fill(COLOR_BLACK)
    screen_w, screen_h = screen.get_size()
    if screen_w == 0 or screen_h == 0: return
    logical_aspect = LOGICAL_WIDTH / LOGICAL_HEIGHT;
    screen_aspect = screen_w / screen_h
    if screen_aspect > logical_aspect:
        new_h = screen_h; new_w = int(new_h * logical_aspect)
    else:
        new_w = screen_w; new_h = int(new_w / logical_aspect)
    scaled_surface = pygame.transform.smoothscale(game_surface, (new_w, new_h))
    blit_pos = ((screen_w - new_w) // 2, (screen_h - new_h) // 2)
    screen.blit(scaled_surface, blit_pos)


# --- MAIN LOOP ---
running = True
while running:
    temps_actuel = time.time()
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.VIDEORESIZE:
            if not is_fullscreen: screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: running = False
            if event.key == pygame.K_F11:
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                else:
                    screen = pygame.display.set_mode((LOGICAL_WIDTH, LOGICAL_HEIGHT), pygame.RESIZABLE)

    if etat_du_jeu_actuel == ETAT_MENU:
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: gerer_clic_menu(event.pos)
            if event.type == pygame.KEYDOWN:
                if saisie_active == 'gamma':
                    if event.key == pygame.K_BACKSPACE:
                        texte_saisie_gamma = texte_saisie_gamma[:-1]
                    elif event.unicode.isdigit() or (event.unicode == '.' and '.' not in texte_saisie_gamma):
                        texte_saisie_gamma += event.unicode
                elif isinstance(saisie_active, str) and saisie_active.startswith("proba_"):
                    direction = int(saisie_active.split('_')[1])
                    if event.key == pygame.K_BACKSPACE:
                        textes_saisie_probas[direction] = textes_saisie_probas[direction][:-1]
                    elif event.unicode.isdigit() or (
                            event.unicode == '.' and '.' not in textes_saisie_probas[direction]):
                        textes_saisie_probas[direction] += event.unicode
        dessiner_menu()
    elif etat_du_jeu_actuel == ETAT_JEU:
        action_a_prendre = None
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                logical_pos = get_logical_pos(event.pos)
                if slider_rect and handle_rect and slider_rect.inflate(0, 20).collidepoint(
                    logical_pos): slider_is_dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                slider_is_dragging = False
            elif event.type == pygame.MOUSEMOTION and slider_is_dragging:
                logical_pos = get_logical_pos(event.pos)
                if slider_rect.width > 0: percent = max(0, min(1, (logical_pos[
                                                                       0] - slider_rect.x) / slider_rect.width)); AUTO_MODE_DELAY_JEU = MAX_DELAY - percent * (
                            MAX_DELAY - MIN_DELAY)
            elif event.type == pygame.KEYDOWN:
                if reset_episode_needed and event.key not in [pygame.K_m, pygame.K_F11,
                                                              pygame.K_ESCAPE]: reset_episode_pygame(); continue
                if event.key == pygame.K_m: etat_du_jeu_actuel = ETAT_MENU; continue
                if config_mode_jeu_initial == "Libre" and not reset_episode_needed:
                    action_map = {pygame.K_UP: 0, pygame.K_DOWN: 1, pygame.K_LEFT: 2, pygame.K_RIGHT: 3}
                    if event.key in action_map: action_a_prendre = action_map[
                        event.key]; dernier_message = f"Manuel: {action_idx_en_fleche.get(action_a_prendre, '?')}"
        is_auto_mode = config_mode_jeu_initial in ["Auto", "AgentPassifTD", "Agent Actif (Q-L)"];
        if is_auto_mode and not reset_episode_needed and (
                temps_actuel - temps_derniere_action_auto > AUTO_MODE_DELAY_JEU):
            if config_mode_jeu_initial == "Agent Actif (Q-L)":
                s_prime, r = agent_etat_idx, config_recompense_non_terminal
                if s_prime in indices_etats_terminaux: r += indices_etats_terminaux[s_prime]
                if q_learning_s is not None:
                    Nsa_table[q_learning_s, q_learning_a] += 1;
                    alpha = 60 / (59 + Nsa_table[q_learning_s, q_learning_a]);
                    q_prime_max = np.max(Q_table[s_prime]) if s_prime not in indices_etats_terminaux else 0;
                    update = alpha * (r + config_gamma * q_prime_max - Q_table[q_learning_s, q_learning_a]);
                    Q_table[q_learning_s, q_learning_a] += update
                if s_prime in indices_etats_terminaux:
                    reset_episode_pygame()
                else:
                    if random.random() < epsilon_q_learning:
                        action_a_prendre = random.randint(0, nb_action - 1)
                    else:
                        action_a_prendre = np.argmax(Q_table[s_prime])
                    q_learning_s, q_learning_a = s_prime, action_a_prendre;
                    dernier_message = f"Agent Q-L: {action_idx_en_fleche.get(action_a_prendre, '?')}";
                    temps_derniere_action_auto = temps_actuel
            elif agent_etat_idx not in indices_etats_terminaux:
                action_auto = pi_politique_visualisee_et_auto[agent_etat_idx]
                if config_mode_jeu_initial == "AgentPassifTD":
                    nb_passages_etat_td[agent_etat_idx] = nb_passages_etat_td.get(agent_etat_idx, 0) + 1;
                    alpha = 1.0 / nb_passages_etat_td[agent_etat_idx];
                    s_prime, r = passer_etat_suivant_env_pygame(agent_etat_idx, action_auto);
                    cible = r + config_gamma * valeurs_etat_a_afficher[s_prime];
                    valeurs_etat_a_afficher[agent_etat_idx] += alpha * (
                                cible - valeurs_etat_a_afficher[agent_etat_idx]);
                    dernier_message = f"Agent TD: {action_idx_en_fleche.get(action_auto, '?')}"
                else:
                    dernier_message = f"Auto: {action_idx_en_fleche.get(action_auto, '?')}"
                action_a_prendre = action_auto;
                temps_derniere_action_auto = temps_actuel
            else:
                reset_episode_pygame()

        if action_a_prendre is not None and not reset_episode_needed:
            s_actuel = agent_etat_idx
            s_prime, r_transition = passer_etat_suivant_env_pygame(s_actuel, action_a_prendre)

            if s_prime == s_actuel:
                shake_timer = HURT_DURATION
                hurt_timer = HURT_DURATION
                is_agent_hurt = True

            agent_etat_idx = s_prime

            reward_for_this_step = 0
            is_terminal = agent_etat_idx in indices_etats_terminaux

            if is_terminal:
                reward_for_this_step = indices_etats_terminaux[agent_etat_idx]
            else:
                reward_for_this_step = r_transition

            score_total_episode_actuel += (config_gamma ** current_episode_step) * reward_for_this_step

            if not is_terminal:
                current_episode_step += 1
            else:
                val_term = indices_etats_terminaux[agent_etat_idx]
                msg_term = f"GAGNÉ ({val_term:+})!" if val_term > 0 else f"PERDU ({val_term})!"
                partie_terminee_message = f"{msg_term} Score: {score_total_episode_actuel:.2f}."
                dernier_message = msg_term
                reset_episode_needed = True
                if is_auto_mode:
                    time.sleep(0.5)
                    reset_episode_pygame()

        dt = clock.get_time() / 1000.0
        if shake_timer > 0: shake_timer -= dt
        if hurt_timer > 0:
            hurt_timer -= dt
            if hurt_timer <= 0: is_agent_hurt = False
        draw_game_view()
    elif etat_du_jeu_actuel == ETAT_EDITER_POLITIQUE:
        for event in events: gerer_evenements_editeur_politique(event)
        dessiner_editeur_politique()
    elif etat_du_jeu_actuel == ETAT_EDITER_GRILLE:
        for event in events: gerer_evenements_editeur_grille(event)
        dessiner_editeur_grille()

    render_to_screen()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()