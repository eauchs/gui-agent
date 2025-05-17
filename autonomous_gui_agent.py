import json
import math # Non utilisé directement, mais pourrait l'être
import os
import time
import tkinter as tk
from tkinter import Label # tk.Label est plus conventionnel si tk est importé comme 'tk'
import pyautogui
from PIL import Image, ImageGrab # Pillow
from pynput.mouse import Controller as MouseController
from rich import print as rich_print
from rich.prompt import Prompt
import base64
import io
import re
import unicodedata
import numpy as np # S'assurer que numpy est installé
import logging

# --- Configuration du Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Chargement de la Configuration (Variables d'Environnement avec Fallbacks) ---
# IMPORTANT: Assurez-vous que ces noms de modèles correspondent EXACTEMENT
# aux modèles disponibles et chargés sur votre serveur local (ex: LM Studio).
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", "http://localhost:1234/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio") # Clé API factice pour serveurs locaux
VLM_MODEL_NAME_FOR_API = os.getenv("VLM_MODEL_NAME_FOR_API", "internvl3-8b-instruct") # Ex: Nom du modèle VLM chargé
QWEN_MODEL_NAME_FOR_API = os.getenv("QWEN_MODEL_NAME_FOR_API", "qwen/qwen3-8b") # Ex: Nom du modèle Qwen chargé

# --- Configuration des Constantes ---
OVERLAY_DURATION = 0.8        # Durée par défaut de l'overlay d'action
HIGHLIGHT_DURATION = 0.15     # Durée de surbrillance du clic
CURSOR_ANIMATION_DURATION = 0.2 # Durée de l'animation du curseur
MAX_AGENT_STEPS = 20          # Nombre maximum d'étapes par tâche globale
MAX_CONSECUTIVE_VLM_FAILURES_BEFORE_QWEN_MODIFIES = 2 # Seuil pour que Qwen intervienne plus directement

# Initialisation des contrôleurs et des librairies
# Client OpenAI (ou compatible)
try:
    from openai import OpenAI
    # S'assurer que le client est initialisé pour être utilisé globalement si nécessaire
    client = OpenAI(base_url=OPENAI_API_BASE_URL, api_key=OPENAI_API_KEY)
except ImportError:
    logging.critical("La librairie OpenAI Python n'est pas installée. Veuillez l'installer avec 'pip install openai'")
    rich_print("[bold red]Erreur Fatale: Librairie OpenAI non trouvée. Exécutez: pip install openai[/bold red]")
    exit()
except Exception as e:
    logging.critical(f"Erreur Fatale: Impossible d'initialiser le client OpenAI: {e}")
    rich_print(f"[bold red]Erreur Fatale: Impossible d'initialiser le client OpenAI: {e}[/bold red]")
    exit()

# Modules Audio
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_ENABLED = True
except ImportError:
    logging.warning("sounddevice ou soundfile non trouvé. Le retour audio sera désactivé.")
    rich_print("[yellow]Attention: sounddevice ou soundfile non trouvé. Le retour audio sera désactivé.[/yellow]")
    AUDIO_ENABLED = False

# Résolution d'écran
try:
    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
except Exception as e: # Peut échouer dans certains environnements (ex: headless, VM sans GUI)
    logging.error(f"Erreur lors de la récupération de la taille de l'écran via pyautogui: {e}")
    rich_print(f"[red]Erreur PyAutoGUI pour taille écran: {e}.[/red]")
    logging.warning("Utilisation par défaut de 1920x1080. Ajustez si nécessaire.")
    rich_print("[yellow]Utilisation par défaut de 1920x1080. Ajustez si nécessaire.[/yellow]")
    SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

# Contrôleur de souris Pynput
try:
    pynput_mouse_controller = MouseController()
except Exception as e:
    logging.warning(f"Impossible d'initialiser Pynput MouseController: {e}. Certaines animations de curseur pourraient être moins fluides.")
    pynput_mouse_controller = None # Fallback

# --- Prompt Système pour le VLM Frontend (Perception) ---
VLM_SYSTEM_PROMPT = f"""
You are a VLM assistant for a macOS GUI agent. You analyze screenshots and follow SPECIFIC INSTRUCTIONS from a supervisor LLM.
Your screen resolution is {SCREEN_WIDTH}x{SCREEN_HEIGHT}.
YOUR ENTIRE RESPONSE MUST BE A SINGLE, VALID JSON OBJECT. NO OTHER TEXT BEFORE OR AFTER THE JSON OBJECT.

The JSON object structure MUST be: {{ "global_thought": {{...}}, "action_sequence": [...] }}

"global_thought" MUST BE A JSON OBJECT. ALL "global_thought" keys listed below ARE MANDATORY. If a key's value is unknown or not applicable, YOU MUST use "N/A" for its value. DO NOT OMIT ANY KEYS.
1.  "Current State Summary": Your summary of task progress for the current instruction.
2.  "User's Current Instruction": Restate the specific instruction you are working on.
3.  "Previous Action Assessment": Outcome of YOUR VLM's PREVIOUS action_sequence for THIS instruction (e.g., "Successfully opened Spotlight", "Clicked button, new dialog appeared", "N/A if first VLM step for this instruction").
4.  "Current Screen Analysis (Brief)": Describe the screenshot relevant to the CURRENT instruction.
5.  "Next Immediate Sub-goal for THIS Instruction": Your specific sub-goal for THIS instruction.
6.  "Action Justification & Selection": Justify your proposed 'action_sequence' for THIS sub-goal.
7.  "Anticipated Next Step AFTER THIS sequence": What you expect on screen after your actions, related to THIS instruction.

The "action_sequence" key MUST ALWAYS BE PRESENT, and its value MUST BE A LIST of micro-action objects. If no actions are needed for the current instruction (e.g., a confirmation task where the state is already as expected), "action_sequence" MUST be an empty list [].
Available "action_type" for micro-actions (ensure all parameters are included as specified):
"CLICK" ({{ "action_type": "CLICK", "position": [norm_x, norm_y], "description": "..." }}),
"DOUBLE_CLICK" ({{ "action_type": "DOUBLE_CLICK", "position": [norm_x, norm_y], "description": "..." }}),
"INPUT" ({{ "action_type": "INPUT", "value": "text_to_type", "position": [opt_norm_x, opt_norm_y], "description": "..." }}), # 'position' is optional, 'value' is required.
"SCROLL" ({{ "action_type": "SCROLL", "direction": "up/down", "description": "..." }}), # 'direction' is required.
"PRESS_ENTER" ({{ "action_type": "PRESS_ENTER", "description": "..." }}),
"KEY_PRESS" ({{ "action_type": "KEY_PRESS", "keys": ["MODIFIER_IF_ANY", "KEY"], "description": "..." }}), # 'keys' MUST be a list of strings. Example: {{"action_type": "KEY_PRESS", "keys": ["COMMAND", "SPACE"], "description": "Open Spotlight"}}
"PAUSE" ({{ "action_type": "PAUSE", "duration_seconds": float_value, "description": "..." }}), # 'duration_seconds' is required.
"FINISHED" ({{ "action_type": "FINISHED", "reason": "State why the instruction is complete.", "description": "Instruction fully completed." }}) -> Use this if your CURRENT INSTRUCTION is fully completed by the proposed actions or current screen state.

Focus ONLY on the immediate instruction. The supervisor LLM (Qwen) handles the overall user goal.
If the instruction is to "confirm if X is visible", and X is visible, your "action_sequence" might be an empty list [], and "global_thought" should reflect this confirmation in "Current Screen Analysis (Brief)" or "Current State Summary". If X is not visible and the instruction was just to check, still use an empty list for actions if no action is to be taken to make it visible.
"""

# --- Fonctions Audio ---
def play_sound_feedback(sound_file_name):
    if not AUDIO_ENABLED: return
    try:
        audio_dir = "audio_feedback"
        os.makedirs(audio_dir, exist_ok=True)
        file_path = os.path.join(audio_dir, sound_file_name)
        if os.path.exists(file_path):
            data, fs = sf.read(file_path, dtype='float32')
            sd.play(data, fs)
        else:
            logging.warning(f"Fichier audio non trouvé: {file_path}")
            rich_print(f"[yellow]Fichier audio non trouvé: {file_path}[/yellow]")
    except Exception as e:
        logging.warning(f"Impossible de jouer le son {sound_file_name}: {e}")
        rich_print(f"[yellow]Impossible de jouer le son {sound_file_name}: {e}[/yellow]")

# --- Fonctions Utilitaires pour l'Overlay et l'Animation du Curseur ---
def animate_cursor_movement(start_x, start_y, end_x, end_y, duration=CURSOR_ANIMATION_DURATION):
    steps = max(1, int(duration * 100))
    for i in range(steps + 1):
        t = i / steps
        current_x = start_x + (end_x - start_x) * t
        current_y = start_y + (end_y - start_y) * t
        try:
            if pynput_mouse_controller:
                pynput_mouse_controller.position = (current_x, current_y)
            else: # Fallback si pynput n'est pas dispo
                pyautogui.moveTo(current_x, current_y, duration=0) # Mouvement instantané pour chaque pas
        except Exception: # Fallback plus large
             pyautogui.moveTo(current_x, current_y, duration=duration / steps / 20 if steps > 0 else 0)
        if steps > 0 and duration > 0:
            time.sleep(duration / steps)

def highlight_click_position(x, y, duration=HIGHLIGHT_DURATION):
    current_pos_x, current_pos_y = pyautogui.position()
    animate_cursor_movement(current_pos_x, current_pos_y, x, y)
    try:
        if pynput_mouse_controller:
            pynput_mouse_controller.position = (x,y)
        else:
            pyautogui.moveTo(x,y, duration=0)
    except Exception:
         pyautogui.moveTo(x,y, duration=0)
    time.sleep(duration)

def create_action_overlay(action_text, x, y, color="lime", duration=OVERLAY_DURATION):
    overlay_root = None
    try:
        overlay_root = tk.Tk()
        overlay_root.overrideredirect(True)
        overlay_root.attributes("-topmost", True)
        overlay_root.attributes("-alpha", 0.75) # Transparence
        overlay_root.config(bg="black") # Couleur de fond pour contraste
        
        lines = action_text.split('\n')
        # Estimations grossières pour la taille, ajustez si nécessaire
        font_avg_width = 8  # Largeur moyenne d'un caractère
        font_avg_height = 18 # Hauteur moyenne d'une ligne
        max_line_len = max(len(line) for line in lines) if lines else 0
        
        overlay_width = (max_line_len * font_avg_width) + 40 # Padding horizontal
        overlay_height = (len(lines) * font_avg_height) + 20 # Padding vertical
        
        screen_w_tk, screen_h_tk = overlay_root.winfo_screenwidth(), overlay_root.winfo_screenheight()
        
        # Positionner l'overlay près du clic, mais en s'assurant qu'il reste à l'écran
        final_x = min(max(0, x + 25), screen_w_tk - overlay_width)
        final_y = min(max(0, y + 25), screen_h_tk - overlay_height)
        
        overlay_root.geometry(f"{int(overlay_width)}x{int(overlay_height)}+{int(final_x)}+{int(final_y)}")
        
        label = tk.Label(overlay_root, text=action_text, fg=color, bg="black", font=("Arial", 12, "bold"), justify=tk.LEFT)
        label.pack(padx=10, pady=5, expand=True, fill='both')
        
        overlay_root.after(int(duration * 1000), lambda r=overlay_root: safe_destroy_overlay(r))
        overlay_root.update() # Forcer l'affichage
        return overlay_root # Retourner la référence pour une destruction potentielle plus tôt
    except tk.TclError as e: # Erreurs Tkinter spécifiques
        logging.warning(f"Erreur TclError Overlay (peut arriver si Tk est mal initialisé): {e}")
        if overlay_root: safe_destroy_overlay(overlay_root) # Tentative de nettoyage
        return None
    except Exception as e:
        logging.warning(f"Avertissement création Overlay: {e}")
        if overlay_root: safe_destroy_overlay(overlay_root) # Tentative de nettoyage
        return None

def safe_destroy_overlay(overlay_ref):
    try:
        if overlay_ref and overlay_ref.winfo_exists(): # Vérifier si la fenêtre existe encore
            overlay_ref.destroy()
    except tk.TclError as e: # Souvent si déjà détruit ou application parente fermée
        logging.debug(f"Erreur TclError bénigne lors de la destruction de l'overlay: {e}")
    except Exception as e:
        logging.warning(f"Erreur inattendue lors de la destruction de l'overlay: {e}")

# --- Fonctions d'Action GUI ---
def action_click(position_norm, description=""):
    x, y = round(position_norm[0] * SCREEN_WIDTH), round(position_norm[1] * SCREEN_HEIGHT)
    logging.info(f"Exécution: CLICK à ({x}, {y}) Description: {description}")
    rich_print(f"Exécution: CLICK à ({x}, {y}) Description: {description}")
    overlay = create_action_overlay(f"CLICK\n({x},{y})\n{description[:30]}", x, y)
    action_successful = False
    try:
        highlight_click_position(x, y)
        pyautogui.click(x=x, y=y)
        action_successful = True
    except Exception as e:
        logging.error(f"Erreur lors de action_click: {e}")
    finally: # S'assurer que l'overlay est détruit même si l'action échoue
        if overlay:
             # L'overlay a son propre timer, mais on peut le forcer ici après un court délai
            overlay.after(200, lambda r=overlay: safe_destroy_overlay(r))
        return action_successful


def action_double_click(position_norm, description=""):
    x, y = round(position_norm[0] * SCREEN_WIDTH), round(position_norm[1] * SCREEN_HEIGHT)
    logging.info(f"Exécution: DOUBLE_CLICK à ({x}, {y}) Description: {description}")
    rich_print(f"Exécution: DOUBLE_CLICK à ({x}, {y}) Description: {description}")
    overlay = create_action_overlay(f"DBL_CLICK\n({x},{y})\n{description[:30]}", x, y)
    action_successful = False
    try:
        highlight_click_position(x, y)
        pyautogui.doubleClick(x=x, y=y, interval=0.1)
        action_successful = True
    except Exception as e:
        logging.error(f"Erreur lors de action_double_click: {e}")
    finally:
        if overlay:
            overlay.after(200, lambda r=overlay: safe_destroy_overlay(r))
        return action_successful

def action_input_text(value_to_type, position_norm=None, description=""):
    logging.info(f"Exécution: INPUT '{value_to_type}' Description: {description}")
    rich_print(f"Exécution: INPUT '{value_to_type}' Description: {description}")
    overlay_x, overlay_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    overlay = None
    action_successful = False
    try:
        if position_norm and isinstance(position_norm, list) and len(position_norm) == 2:
            x_click, y_click = round(position_norm[0] * SCREEN_WIDTH), round(position_norm[1] * SCREEN_HEIGHT)
            logging.info(f"Clic optionnel à ({x_click}, {y_click}) avant de taper.")
            rich_print(f"Clic optionnel à ({x_click}, {y_click}) avant de taper.")
            highlight_click_position(x_click, y_click, duration=0.1) # Plus court pour clic avant input
            pyautogui.click(x=x_click, y=y_click)
            time.sleep(0.2) # Délai pour que le focus se fasse
            overlay_x, overlay_y = x_click, y_click
        else:
            logging.info("Aucune position de clic spécifiée pour INPUT, frappe directe.")
            rich_print("Aucune position de clic spécifiée pour INPUT, frappe directe.")

        overlay_text = f"TYPE:\n{value_to_type[:25]}{'...' if len(value_to_type) > 25 else ''}\n{description[:30]}"
        overlay_duration = max(0.5, min(len(value_to_type) * 0.05, 3.0)) # Durée proportionnelle
        overlay = create_action_overlay(overlay_text, overlay_x, overlay_y, duration=overlay_duration)
        
        pyautogui.write(value_to_type, interval=0.03)
        action_successful = True
    except Exception as e:
        logging.error(f"Erreur lors de action_input_text: {e}")
    finally:
        # L'overlay créé par create_action_overlay se détruira via son propre timer.
        # Si on veut une destruction immédiate après pyautogui.write, il faudrait appeler safe_destroy_overlay(overlay) ici.
        # Pour l'instant, on se fie à son timer interne.
        return action_successful

def action_scroll(direction, description=""):
    scroll_clicks = 10 # Ajuster pour sensibilité
    amount = -scroll_clicks if direction.lower() == "up" else scroll_clicks
    logging.info(f"Exécution: SCROLL {direction.upper()} Description: {description}")
    rich_print(f"Exécution: SCROLL {direction.upper()} Description: {description}")
    overlay = create_action_overlay(f"SCROLL {direction.upper()}\n{description[:30]}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    action_successful = False
    try:
        pyautogui.scroll(amount)
        action_successful = True
    except Exception as e:
        logging.error(f"Erreur lors de action_scroll: {e}")
    finally:
        if overlay:
            overlay.after(200, lambda r=overlay: safe_destroy_overlay(r))
        return action_successful

def action_press_enter(description=""):
    logging.info(f"Exécution: PRESS_ENTER Description: {description}")
    rich_print(f"Exécution: PRESS_ENTER Description: {description}")
    overlay = create_action_overlay(f"ENTRÉE\n{description[:30]}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    action_successful = False
    try:
        pyautogui.press("enter")
        action_successful = True
    except Exception as e:
        logging.error(f"Erreur lors de action_press_enter: {e}")
    finally:
        if overlay:
            overlay.after(200, lambda r=overlay: safe_destroy_overlay(r))
        return action_successful

KEY_MAP = { # Étendu et normalisé
    "command": "command", "cmd": "command", "super": "command", "win": "command", # Mac/Win common term for Super
    "option": "alt",  # PyAutoGUI uses 'alt' for macOS Option key
    "alt": "alt",
    "control": "ctrl", "ctrl": "ctrl",
    "shift": "shift",
    "space": "space", "espace": "space",
    "delete": "delete", # Sur Mac, 'delete' est backspace. Pour forward delete, c'est 'fn'+'delete'.
                        # Pour PyAutoGUI, 'delete' est souvent forward delete, 'backspace' est backspace.
                        # Clarifions: 'delete' -> forward delete, 'backspace' -> backspace
    "backspace": "backspace", "effacer": "backspace",
    "enter": "enter", "return": "enter", "entrée": "enter",
    "tab": "tab", "tabulation": "tab",
    "escape": "esc", "esc": "esc", "échap": "esc",
    "up_arrow": "up", "flèche_haut": "up", "haut": "up", "up": "up",
    "down_arrow": "down", "flèche_bas": "down", "bas": "down", "down": "down",
    "left_arrow": "left", "flèche_gauche": "left", "gauche": "left", "left": "left",
    "right_arrow": "right", "flèche_droite": "right", "droite": "right", "right": "right",
    # F-keys
    **{f"f{i}": f"f{i}" for i in range(1, 13)},
    # Autres touches spéciales
    "pageup": "pageup", "pagedown": "pagedown",
    "home": "home", "end": "end",
    "insert": "insert",
    "printscreen": "printscreen", "prtscn": "printscreen",
    "capslock": "capslock", "numlock": "numlock", "scrolllock": "scrolllock",
}

def action_key_press(keys_to_press_list, description=""):
    logging.info(f"Tentative KEY_PRESS: {keys_to_press_list}. Description: {description}")
    rich_print(f"Tentative KEY_PRESS: {keys_to_press_list}. Description: {description}")
    ov = None
    action_successful = False
    try:
        if not isinstance(keys_to_press_list, list) or not keys_to_press_list:
            logging.error(f"Erreur KEY_PRESS: 'keys' doit être une liste non vide. Reçu: {keys_to_press_list}")
            rich_print(f"[red]Erreur KEY_PRESS: 'keys' doit être une liste non vide. Reçu: {keys_to_press_list}[/red]")
            play_sound_feedback("error.wav")
            return False # Retourner False ici

        pyautogui_keys = []
        for key_name in keys_to_press_list:
            key_name_lower = str(key_name).lower().strip()
            mapped_key = KEY_MAP.get(key_name_lower)
            
            if mapped_key:
                pyautogui_keys.append(mapped_key)
            # Pour les caractères alphanumériques et symboles courants non explicitement dans KEY_MAP
            elif len(key_name_lower) == 1 and (key_name_lower.isalnum() or key_name_lower in pyautogui.KEYBOARD_KEYS_FUNCTION_KEYS_SYMBOLS): # Utilise la liste de pyautogui
                pyautogui_keys.append(key_name_lower)
            else:
                logging.warning(f"Touche inconnue ou non gérée '{key_name_lower}' dans KEY_PRESS. Ignorée. Original: {keys_to_press_list}")
                rich_print(f"[red]Attention: Touche inconnue ou non gérée '{key_name_lower}' dans KEY_PRESS. Ignorée. Original: {keys_to_press_list}[/red]")
        
        if not pyautogui_keys:
            logging.error(f"Erreur KEY_PRESS: Aucune touche PyAutoGUI valide après normalisation. Original VLM: {keys_to_press_list}")
            rich_print(f"[red]Erreur KEY_PRESS: Aucune touche PyAutoGUI valide à presser après normalisation. Original VLM: {keys_to_press_list}[/red]")
            play_sound_feedback("error.wav")
            return False # Retourner False ici

        logging.info(f"Exécution: KEY_PRESS PyAutoGUI: {pyautogui_keys} (Original VLM: {keys_to_press_list})")
        rich_print(f"Exécution: KEY_PRESS PyAutoGUI: {pyautogui_keys} (Original VLM: {keys_to_press_list})")
        ov = create_action_overlay(f"RACCOURCI:\n{'+'.join(map(str,pyautogui_keys))}\n{description[:30]}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, color="orange")

        # Gestion améliorée pour hotkey vs pressions séquentielles
        potential_modifiers = {"command", "ctrl", "alt", "shift", "option"} # 'option' est mappé sur 'alt'
        
        # Cas typique de raccourci: un ou plusieurs modificateurs suivis d'une touche non-modificatrice
        if len(pyautogui_keys) > 1 and pyautogui_keys[-1] not in potential_modifiers and \
           all(k in potential_modifiers for k in pyautogui_keys[:-1]):
            pyautogui.hotkey(*pyautogui_keys)
        elif len(pyautogui_keys) == 1: # Une seule touche
            if pyautogui_keys[0] in potential_modifiers:
                 logging.warning(f"KEY_PRESS: Pression d'une seule touche modificatrice '{pyautogui_keys[0]}'. Ceci n'a généralement pas d'effet seul.")
                 rich_print(f"[yellow]KEY_PRESS: Pression d'une seule touche modificatrice '{pyautogui_keys[0]}' demandée. Ceci n'a généralement pas d'effet seul.[/yellow]")
            pyautogui.press(pyautogui_keys[0])
        else: # Séquence complexe ou multiple touches non-modificatrices
            logging.warning(f"Séquence de touches complexe/non standard pour hotkey ({pyautogui_keys}), tentative de pressions individuelles maintenues si applicable.")
            rich_print(f"[yellow]Séquence de touches complexe/non standard ({pyautogui_keys}), tentative de pressions individuelles maintenues.[/yellow]")
            
            modifiers_down = [k for k in pyautogui_keys if k in potential_modifiers]
            main_keys = [k for k in pyautogui_keys if k not in potential_modifiers]

            for mod in modifiers_down: pyautogui.keyDown(mod)
            time.sleep(0.05) # Petit délai
            for key in main_keys: pyautogui.press(key)
            time.sleep(0.05) # Petit délai
            for mod in reversed(modifiers_down): pyautogui.keyUp(mod) # Relâcher dans l'ordre inverse
        
        action_successful = True
    except Exception as e:
        logging.critical(f"Erreur durant KEY_PRESS: {e}")
        rich_print(f"[bold red]KEY_PRESS Error: {e}[/bold red]") # Utiliser rich_print
        play_sound_feedback("error.wav")
        # action_successful reste False
    finally:
        if ov: ov.after(200, lambda r=ov: safe_destroy_overlay(r)) # Assurer que ov est défini
        return action_successful


def action_pause(duration_seconds, description=""):
    logging.info(f"Exécution: PAUSE pour {duration_seconds}s Description: {description}")
    rich_print(f"Exécution: PAUSE pour {duration_seconds}s Description: {description}")
    overlay = create_action_overlay(f"PAUSE {duration_seconds}s\n{description[:30]}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, color="blue")
    action_successful = False
    try:
        duration_val = float(duration_seconds)
        if duration_val < 0: duration_val = 0
        time.sleep(duration_val)
        action_successful = True
    except ValueError:
        logging.error(f"Erreur PAUSE: Durée invalide '{duration_seconds}'. Utilisation de 0s.")
        rich_print(f"[red]Erreur PAUSE: Durée invalide '{duration_seconds}'. Utilisation de 0s.[/red]")
        time.sleep(0)
    except Exception as e:
        logging.error(f"Erreur inattendue durant PAUSE: {e}")
    finally:
        if overlay: safe_destroy_overlay(overlay) # Laisser create_action_overlay gérer sa propre durée principale
        return action_successful

def action_finished_vlm(reason="VLM: L'instruction semble terminée.", description=""):
    logging.info(f"Indication VLM: FINISHED. Raison VLM: {reason} Description: {description}")
    rich_print(f"[green]Action VLM: FINISHED. Raison VLM:[/green] {reason} (Desc: {description})")
    play_sound_feedback("ok.wav")
    return True # Cette action elle-même réussit toujours, son impact est logique.

# --- Fonctions de l'Agent ---
def image_to_base64_url(image_path_or_obj, target_format="PNG"): # Renommé format -> target_format
    try:
        img = None
        if isinstance(image_path_or_obj, str):
            with Image.open(image_path_or_obj) as loaded_img:
                img = loaded_img.copy()
        elif isinstance(image_path_or_obj, Image.Image):
            img = image_path_or_obj
        else:
            raise ValueError("image_path_or_obj doit être un chemin (str) ou un objet PIL.Image.")

        # Assurer la conversion en RGB pour les formats qui ne supportent pas l'alpha (comme JPEG)
        if target_format.upper() == "JPEG" and (img.mode == 'RGBA' or img.mode == 'LA' or img.mode == 'P'):
            img = img.convert('RGB')
        
        buffered = io.BytesIO()
        img.save(buffered, format=target_format) # Utiliser target_format
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/{target_format.lower()};base64,{base64_str}"
    except Exception as e:
        logging.error(f"Erreur d'encodage de l'image en base64: {e}")
        rich_print(f"[red]Erreur d'encodage image: {e}[/red]")
        return None

def parse_vlm_output_to_sequence(vlm_response_str: str):
    json_str_to_parse = None
    cleaned_str = vlm_response_str.strip()

    match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", cleaned_str, re.DOTALL)
    if match:
        json_str_to_parse = match.group(1)
        logging.debug("JSON VLM extrait d'un bloc Markdown.")
    else:
        first_brace = cleaned_str.find('{')
        last_brace = cleaned_str.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            json_str_to_parse = cleaned_str[first_brace : last_brace + 1]
            logging.debug("JSON VLM extrait par première/dernière accolade.")
        else:
            logging.error(f"Aucune structure JSON identifiable trouvée dans la réponse VLM: {cleaned_str[:200]}...")
            rich_print(f"[red]Erreur VLM: Aucune structure JSON identifiable trouvée: {cleaned_str[:200]}...[/red]")
            return None
    try:
        cleaned_for_parsing = "".join(ch for ch in json_str_to_parse if unicodedata.category(ch)[0]!="C" or ch in ('\t','\n','\r'))
        decoded_data = json.loads(cleaned_for_parsing)

        if not isinstance(decoded_data, dict):
            logging.error(f"Le JSON VLM décodé n'est pas un dictionnaire. Type: {type(decoded_data)}")
            rich_print(f"[red]Erreur VLM: JSON décodé n'est pas un dictionnaire. Type: {type(decoded_data)}[/red]")
            return None

        global_thought_raw = decoded_data.get("global_thought")
        action_sequence_raw = decoded_data.get("action_sequence")

        if not isinstance(global_thought_raw, dict): # Doit être un objet, même si vide (non idéal)
            logging.error(f"'global_thought' n'est pas un objet JSON ou est manquant. Reçu: {global_thought_raw}")
            rich_print(f"[red]Erreur VLM: 'global_thought' n'est pas un objet JSON valide. Reçu: {str(global_thought_raw)[:100]}[/red]")
            return None

        parsed_global_thought = {}
        expected_thought_keys = [
            "Current State Summary", "User's Current Instruction", "Previous Action Assessment",
            "Current Screen Analysis (Brief)", "Next Immediate Sub-goal for THIS Instruction",
            "Action Justification & Selection", "Anticipated Next Step AFTER THIS sequence"
        ]
        incomplete_thought = False
        for key in expected_thought_keys:
            value = global_thought_raw.get(key)
            if value is None: # Clé manquante
                logging.warning(f"'global_thought' manque la clé '{key}'. Utilisation de 'N/A'.")
                rich_print(f"[yellow]Avertissement VLM: 'global_thought' manque la clé '{key}'. Utilisation de 'N/A'.[/yellow]")
                parsed_global_thought[key] = "N/A (non fourni par VLM ou null)"
                incomplete_thought = True
            else:
                 parsed_global_thought[key] = value
        
        if not isinstance(action_sequence_raw, list): # Doit être une liste, même vide
            logging.error(f"'action_sequence' n'est pas une liste ou est manquante. Reçu: {action_sequence_raw}")
            rich_print(f"[red]Erreur VLM: 'action_sequence' n'est pas une liste valide ou est manquante. Reçu: {str(action_sequence_raw)[:100]}[/red]")
            return None

        validated_sequence = []
        for i, micro_action in enumerate(action_sequence_raw):
            if not isinstance(micro_action, dict):
                logging.error(f"Micro-action {i} n'est pas un objet JSON. Contenu: {micro_action}")
                rich_print(f"[red]Erreur VLM: Micro-action {i} n'est pas un objet JSON.[/red]")
                return None
            
            action_type = micro_action.get("action_type")
            if not action_type: # action_type est obligatoire
                logging.error(f"Micro-action {i} (Description: {micro_action.get('description','N/A')}) manque 'action_type'.")
                rich_print(f"[red]Erreur VLM: Micro-action {i} manque 'action_type'.[/red]")
                return None
            
            micro_action.setdefault("description", f"Action VLM {i+1} ({action_type}) sans description explicite")


            if action_type in ["CLICK", "DOUBLE_CLICK"]:
                if micro_action.get("position") is None:
                    logging.error(f"'position' requise et manquante/null pour '{action_type}' (micro_action {i}).")
                    rich_print(f"[red]Erreur VLM: 'position' est requise pour {action_type}.[/red]")
                    return None
            # ... (validation plus poussée des champs par action_type, comme dans la version précédente)
            if "position" in micro_action and micro_action["position"] is not None: # Validation et normalisation de la position
                pos = micro_action["position"]
                if not (isinstance(pos, list) and len(pos) == 2 and all(isinstance(p, (float, int, str)) for p in pos)):
                    logging.error(f"Format 'position' invalide pour micro_action {i} ('{action_type}'): {pos}")
                    return None
                try:
                    raw_x, raw_y = float(pos[0]), float(pos[1])
                    norm_x = raw_x / SCREEN_WIDTH if (abs(raw_x) > 1.0001 and SCREEN_WIDTH > 0) else raw_x
                    norm_y = raw_y / SCREEN_HEIGHT if (abs(raw_y) > 1.0001 and SCREEN_HEIGHT > 0) else raw_y
                    micro_action["position"] = [max(0.0, min(1.0, norm_x)), max(0.0, min(1.0, norm_y))]
                except ValueError:
                    logging.error(f"Impossible de convertir position en nombres pour micro_action {i} ('{action_type}'): {pos}")
                    return None
            
            if action_type == "INPUT" and "value" not in micro_action: # 'value' peut être vide, mais la clé doit exister
                 logging.error(f"'value' requise pour 'INPUT' (micro_action {i})."); return None
            if action_type == "SCROLL" and micro_action.get("direction") not in ["up", "down"]:
                 logging.error(f"'direction' (up/down) requise et valide pour 'SCROLL' (micro_action {i}). Reçu: {micro_action.get('direction')}"); return None
            if action_type == "KEY_PRESS" and (not isinstance(micro_action.get("keys"), list) or not micro_action.get("keys")):
                 logging.error(f"'keys' (liste non vide) requise pour 'KEY_PRESS' (micro_action {i})."); return None
            if action_type == "PAUSE":
                if "duration_seconds" not in micro_action:
                     logging.error(f"'duration_seconds' requise pour 'PAUSE' (micro_action {i})."); return None
                try: float(micro_action["duration_seconds"])
                except ValueError: logging.error(f"'duration_seconds' doit être un nombre pour 'PAUSE' (micro_action {i}). Reçu: {micro_action['duration_seconds']}"); return None
            
            validated_sequence.append(micro_action)

        if incomplete_thought:
            logging.info("Le 'global_thought' du VLM était incomplet mais a été traité avec des valeurs par défaut.")
            # rich_print("[grey50]Le 'global_thought' du VLM était incomplet mais a été traité.[/grey50]")

        return {"global_thought": parsed_global_thought, "action_sequence": validated_sequence}

    except json.JSONDecodeError as e:
        error_context_str = cleaned_for_parsing if 'cleaned_for_parsing' in locals() else (json_str_to_parse if json_str_to_parse else cleaned_str)
        logging.error(f"JSONDecodeError VLM: {e}. Sur: '{error_context_str[:300]}...'")
        rich_print(f"[red]Erreur VLM JSONDecodeError: {e}. Tentative sur: '{error_context_str[:300]}...'[/red]")
        return None
    except Exception as e:
        logging.error(f"Erreur inattendue VLM parsing: {e}. Entrée: '{cleaned_str[:200]}...'")
        rich_print(f"[red]Erreur VLM inattendue durant parsing: {e}. Entrée: '{cleaned_str[:200]}...'[/red]")
        return None

def build_messages_for_vlm_api(system_prompt_content, current_instruction_for_vlm, image_base64_url, vlm_execution_history_summary_list):
    user_text_content = f"Current Specific Instruction from Supervisor: {current_instruction_for_vlm}"
    if vlm_execution_history_summary_list: # Limiter la taille de l'historique
        user_text_content += "\n\n--- Summary of Your Previous VLM Steps & Outcomes (for this overall task, max 3 recent shown) ---"
        for i, summary_line in enumerate(vlm_execution_history_summary_list[-3:]):
            user_text_content += f"\n{len(vlm_execution_history_summary_list) - min(3, len(vlm_execution_history_summary_list)) + i + 1}. {summary_line}"
    else:
        user_text_content += "\n\n--- No previous VLM steps for this current VLM instruction phase. ---"

    user_text_content += "\n\n--- Current Situation ---\nAnalyze the screen and follow the 'Current Specific Instruction from Supervisor'. Your ENTIRE response MUST be a single valid JSON object as specified in the system prompt."

    messages = [{"role": "system", "content": system_prompt_content}]
    content_list = [{"type": "text", "text": user_text_content}]
    if image_base64_url:
        content_list.append({"type": "image_url", "image_url": {"url": image_base64_url, "detail": "high"}}) # 'high' pour meilleure analyse
    messages.append({"role": "user", "content": content_list})
    return messages

# --- Prompt Système pour Qwen (Backend Stratégique) ---
QWEN_SYSTEM_PROMPT = f"""
You are an expert strategic supervisor for a macOS GUI automation agent.
The agent has a VLM frontend that analyzes screenshots based on specific instructions you provide, and proposes a GUI 'action_sequence' in JSON.
You (Qwen) are the brain: receive an 'overall_user_goal', give specific instructions to the VLM, analyze the VLM's JSON output (or its failure), and decide the final course of action.
Screen resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}.

You will receive:
- 'overall_user_goal'.
- 'current_vlm_status_report': A dict with 'vlm_output_json_str' (VLM's raw response), 'parsed_vlm_data' (parsed VLM output, or null if parse failed), and 'vlm_error_message' (if VLM failed badly).
- 'image_base64_url': Current screenshot (ONLY if you are a multimodal Qwen-VL model, otherwise ignore this).
- 'interaction_history_summary': Summary of VLM outputs & your past Qwen decisions for the current 'overall_user_goal'.
- 'consecutive_vlm_failures_for_current_instruction_count': How many times VLM has failed (bad JSON or supervisor rejection) for the CURRENT specific VLM instruction.

YOUR TASKS:
1.  CRITICALLY EVALUATE VLM's output (if 'parsed_vlm_data' is available) or its failure.
    - Did VLM follow instructions? Was its 'global_thought' complete and its 'action_sequence' logical?
    - Does VLM's plan realistically help achieve the 'overall_user_goal'? Is it stuck, hallucinating, or in a loop?
    - If VLM proposed "FINISHED", is its specific instruction truly complete in the context of the 'overall_user_goal'?
2.  DECIDE the next course of action by providing a JSON response.

**STRATEGY FOR VLM FAILURES OR INCOMPLETE OUTPUT:**
- If VLM provides unusable JSON or its 'action_sequence' is illogical, or if 'global_thought' is missing critical fields, attempt to "RETRY_VLM_WITH_NEW_INSTRUCTION". Make the new instruction simpler, more direct, or break down the VLM's task.
- If 'consecutive_vlm_failures_for_current_instruction_count' is >= {MAX_CONSECUTIVE_VLM_FAILURES_BEFORE_QWEN_MODIFIES} OR if 'parsed_vlm_data' is consistently null/unusable, you MAY propose your own 'action_sequence_to_execute'.
  Your 'decision_type' for this MUST be "EXECUTE_MODIFIED_SEQUENCE".
  Clearly state in 'reasoning' that you are proposing the sequence due to VLM failure (e.g., "VLM failed {MAX_CONSECUTIVE_VLM_FAILURES_BEFORE_QWEN_MODIFIES} times, attempting direct action: open Spotlight and type X.").
  This 'action_sequence_to_execute' MUST conform to the agent's execution capabilities and use the EXACT SAME format as specified for the VLM (see below). Use with caution.

YOUR ENTIRE RESPONSE MUST BE A SINGLE VALID JSON OBJECT with the following structure. ONLY use the 'decision_type' values listed:
{{
  "decision_type": "[EXECUTE_VLM_SEQUENCE | EXECUTE_MODIFIED_SEQUENCE | RETRY_VLM_WITH_NEW_INSTRUCTION | TASK_COMPLETED | TASK_FAILED]",
  "reasoning": "Your concise explanation for the decision. State if proposing a modified/direct sequence and why.",
  "action_sequence_to_execute": [ /* Array of micro-actions, or null. Required and MUST BE VALID if decision_type is EXECUTE_MODIFIED_SEQUENCE.
                                      If decision_type is EXECUTE_VLM_SEQUENCE, this field will be IGNORED by the agent (it will use VLM's sequence).
                                      Format MUST MATCH VLM's action format EXACTLY:
                                      {{"action_type": "KEY_PRESS", "keys": ["COMMAND", "SPACE"], "description": "Open Spotlight"}},
                                      {{"action_type": "INPUT", "value": "Chrome", "description": "Type Chrome"}}
                                   */ ],
  "next_vlm_instruction": "New specific instruction for VLM, or null. Required if decision_type is RETRY_VLM_WITH_NEW_INSTRUCTION.",
  "user_summary_message": "Optional message for the user if the task is completed or failed."
}}

Available "action_type" for 'action_sequence_to_execute' (WHEN YOU PROVIDE IT under EXECUTE_MODIFIED_SEQUENCE):
"CLICK", "DOUBLE_CLICK", "INPUT", "SCROLL", "PRESS_ENTER", "KEY_PRESS", "PAUSE", "FINISHED".
Use the same parameters as specified for the VLM (e.g., "keys" as a list for KEY_PRESS, "value" for INPUT, "position" for CLICK).
Always aim for progress. Break down complex goals for the VLM.
If VLM's 'global_thought' is missing many fields (filled with "N/A (non fourni par VLM ou null)"), it's struggling; simplify its instruction.
If the 'overall_user_goal' is achieved, set 'decision_type' to "TASK_COMPLETED".
If the task cannot be completed after reasonable attempts, set 'decision_type' to "TASK_FAILED".
DO NOT use any other 'decision_type' than those listed above. DO NOT use 'EXECUTE_BLIND_ACTION'.
"""

VALID_QWEN_DECISION_TYPES = [
    "EXECUTE_VLM_SEQUENCE",
    "EXECUTE_MODIFIED_SEQUENCE",
    "RETRY_VLM_WITH_NEW_INSTRUCTION",
    "TASK_COMPLETED",
    "TASK_FAILED"
]

def get_qwen_strategic_decision(api_client_ref, overall_user_goal, image_base64_url_for_qwen_vl,
                                current_vlm_status_report,
                                full_interaction_history,
                                consecutive_vlm_failures_count):
    qwen_user_prompt_parts = [f"Current Overall User Goal: {overall_user_goal}"]
    vlm_report = current_vlm_status_report
    qwen_user_prompt_parts.append("\n--- VLM Frontend Status for its Last Instruction ---")

    if vlm_report.get("parsed_vlm_data"):
        qwen_user_prompt_parts.append(f"VLM Parsed Output:\n```json\n{json.dumps(vlm_report['parsed_vlm_data'], indent=2, ensure_ascii=False)}\n```")
    else:
        qwen_user_prompt_parts.append(f"VLM FAILED to provide usable/parsable JSON output for its last instruction.")

    if vlm_report.get("vlm_error_message"):
        qwen_user_prompt_parts.append(f"VLM Error/Warning Details: {vlm_report['vlm_error_message']}")
    if not vlm_report.get("parsed_vlm_data") and vlm_report.get("vlm_output_json_str"):
        qwen_user_prompt_parts.append(f"VLM Raw (unparsable/problematic) Output Snippet: ```\n{vlm_report['vlm_output_json_str'][:350]}...\n```")
    
    qwen_user_prompt_parts.append(f"Consecutive VLM Failures for Current VLM Instruction: {consecutive_vlm_failures_count}")

    history_summary_for_qwen_str = "\n\n--- Recent Interaction History (Your Past Qwen Decisions & VLM Outcomes, max 3 recent shown) ---"
    if full_interaction_history:
        for i, (enriched_hist_entry_json_str, _) in enumerate(reversed(full_interaction_history[-3:])):
            try:
                entry_data = json.loads(enriched_hist_entry_json_str)
                qwen_prev_decision = entry_data.get("qwen_decision_obj", {})
                vlm_instr_given_to_vlm = entry_data.get("vlm_instruction_given_to_vlm", "N/A")
                
                summary_line = f"Hist-{i+1}. Your (Qwen) Prev Decision: '{qwen_prev_decision.get('decision_type','N/A')}' for VLM instruction: '{vlm_instr_given_to_vlm}'. "
                summary_line += f"Your Reason: '{qwen_prev_decision.get('reasoning','N/A')}'. "
                
                vlm_status_in_hist = entry_data.get("vlm_status_report_from_that_step", {})
                vlm_parsed_data_in_hist = vlm_status_in_hist.get("parsed_vlm_data", {}) # Nested
                vlm_action_sequence_in_hist = vlm_parsed_data_in_hist.get("action_sequence") if vlm_parsed_data_in_hist else None

                if vlm_action_sequence_in_hist and isinstance(vlm_action_sequence_in_hist, list):
                    action_types_proposed = [a.get('action_type','UNKNOWN') for a in vlm_action_sequence_in_hist if isinstance(a,dict)]
                    summary_line += f" VLM had proposed actions: {action_types_proposed}. "
                elif vlm_status_in_hist.get("vlm_error_message"):
                     summary_line += f" VLM then had error: {vlm_status_in_hist['vlm_error_message']}. "
                
                executed_actions_in_hist = entry_data.get("executed_action_sequence") # From top level of history entry
                if executed_actions_in_hist and isinstance(executed_actions_in_hist, list):
                    action_types_executed = [a.get('action_type','UNKNOWN') for a in executed_actions_in_hist if isinstance(a,dict)]
                    summary_line += f" Actions then executed: {action_types_executed}. "
                elif qwen_prev_decision.get('decision_type') not in ["EXECUTE_VLM_SEQUENCE", "EXECUTE_MODIFIED_SEQUENCE"]:
                     summary_line += f" No direct GUI actions were executed based on Qwen's decision type '{qwen_prev_decision.get('decision_type','N/A')}'. "
                history_summary_for_qwen_str += "\n" + summary_line.strip()
            except Exception as e_hist:
                history_summary_for_qwen_str += f"\nHist-{i+1}. (Erreur de parsing de l'historique pour Qwen: {str(e_hist)[:100]})"
        qwen_user_prompt_parts.append(history_summary_for_qwen_str)
    else:
        qwen_user_prompt_parts.append("\n--- No interaction history yet for this overall task. ---")

    qwen_user_prompt_parts.append("\nBased on all the above, provide your decision in the specified JSON format. Ensure 'decision_type' is one of the allowed values and 'action_sequence_to_execute' (if provided for EXECUTE_MODIFIED_SEQUENCE) uses the correct action format.")
    qwen_messages = [{"role": "system", "content": QWEN_SYSTEM_PROMPT}]
    
    current_user_message_content_list = [{"type": "text", "text": "\n".join(qwen_user_prompt_parts)}]

    is_qwen_multimodal = "VL" in QWEN_MODEL_NAME_FOR_API.upper()
    if is_qwen_multimodal and image_base64_url_for_qwen_vl:
        current_user_message_content_list.append({"type": "image_url", "image_url": {"url": image_base64_url_for_qwen_vl, "detail": "high"}})
        logging.info("Image envoyée au Backend Qwen (car semble multimodal).")
    qwen_messages.append({"role": "user", "content": current_user_message_content_list})

    qwen_response_str_raw = ""
    try:
        logging.info(f"Envoi de la requête au Backend Qwen (Modèle: {QWEN_MODEL_NAME_FOR_API})...")
        rich_print(f"\nEnvoi de la requête au Backend Qwen (Modèle: {QWEN_MODEL_NAME_FOR_API})...")
        completion = api_client_ref.chat.completions.create(
            model=QWEN_MODEL_NAME_FOR_API,
            messages=qwen_messages,
            max_tokens=1800, # Qwen peut être verbeux
            temperature=0.1 # Température basse pour des décisions plus déterministes
        )
        qwen_response_str_raw = completion.choices[0].message.content
        logging.debug(f"Réponse Brute du Backend Qwen:\n{qwen_response_str_raw}")
        rich_print(f"[cyan]Réponse Brute du Backend Qwen:[/]\n{qwen_response_str_raw}")

        json_str_to_parse_qwen = None
        match_qwen = re.search(r"```json\s*(\{[\s\S]+?\})\s*```", qwen_response_str_raw, re.DOTALL)
        if match_qwen:
            json_str_to_parse_qwen = match_qwen.group(1)
        else:
            first_brace_qwen = qwen_response_str_raw.find('{')
            last_brace_qwen = qwen_response_str_raw.rfind('}')
            if first_brace_qwen != -1 and last_brace_qwen > first_brace_qwen:
                json_str_to_parse_qwen = qwen_response_str_raw[first_brace_qwen : last_brace_qwen + 1]
            else:
                raise ValueError(f"Aucune donnée JSON extractible de la réponse de Qwen. Réponse: {qwen_response_str_raw}")
        
        cleaned_json_str_qwen = "".join(ch for ch in json_str_to_parse_qwen if unicodedata.category(ch)[0]!="C" or ch in ('\t','\n','\r'))
        qwen_decision = json.loads(cleaned_json_str_qwen)
        qwen_decision["raw_qwen_response_str_for_debug"] = qwen_response_str_raw

        dt = qwen_decision.get("decision_type")
        if dt not in VALID_QWEN_DECISION_TYPES:
            logging.error(f"Type de décision Qwen invalide: '{dt}'. Reçu: {qwen_decision}")
            raise ValueError(f"Type de décision Qwen invalide: '{dt}'. Valides: {VALID_QWEN_DECISION_TYPES}")

        if dt == "EXECUTE_MODIFIED_SEQUENCE":
            actions = qwen_decision.get("action_sequence_to_execute")
            if not (isinstance(actions, list) and actions): # Doit être une liste non-vide
                logging.warning(f"Qwen: Décision '{dt}' mais 'action_sequence_to_execute' est manquante, vide ou n'est pas une liste. Qwen devrait revoir. Original: {actions}")
                # Convertir en une décision de retry VLM pour éviter une exécution vide
                qwen_decision["decision_type"] = "RETRY_VLM_WITH_NEW_INSTRUCTION"
                qwen_decision["reasoning"] = f"Qwen (self-correction): Tentative EXECUTE_MODIFIED_SEQUENCE avec action_sequence invalide. Passage à RETRY_VLM. Raison initiale: {qwen_decision.get('reasoning')}"
                qwen_decision["next_vlm_instruction"] = "Qwen a tenté de fournir une action mais elle était invalide. Veuillez réévaluer la situation et proposer une nouvelle séquence d'actions VLM."
                qwen_decision["action_sequence_to_execute"] = None
            else: # Valider la structure de chaque action (simplifié ici)
                for i, act in enumerate(actions):
                    if not isinstance(act, dict) or "action_type" not in act:
                        logging.error(f"Qwen: Action {i} dans EXECUTE_MODIFIED_SEQUENCE est mal formatée: {act}")
                        raise ValueError(f"Qwen: Action {i} mal formatée dans EXECUTE_MODIFIED_SEQUENCE.")
        elif dt == "RETRY_VLM_WITH_NEW_INSTRUCTION" and (not qwen_decision.get("next_vlm_instruction") or not str(qwen_decision.get("next_vlm_instruction")).strip()):
            raise ValueError("Décision Qwen 'RETRY_VLM_WITH_NEW_INSTRUCTION' mais 'next_vlm_instruction' est manquante, vide ou non-string.")
        
        # Assurer que les champs non pertinents sont None
        if dt not in ["EXECUTE_MODIFIED_SEQUENCE"]:
            qwen_decision["action_sequence_to_execute"] = None
        if dt != "RETRY_VLM_WITH_NEW_INSTRUCTION":
            qwen_decision["next_vlm_instruction"] = None

        return qwen_decision
        
    except Exception as e:
        logging.critical(f"Erreur Critique durant l'appel Qwen ou parsing: {e}\nRéponse brute Qwen (si disponible): {qwen_response_str_raw}")
        rich_print(f"[bold red]Erreur Critique Qwen: {e}[/bold red]")
        play_sound_feedback("error.wav")
        return { # Fallback robuste
            "decision_type": "TASK_FAILED",
            "reasoning": f"Erreur critique du module stratégique Qwen: {str(e)}",
            "action_sequence_to_execute": None,
            "next_vlm_instruction": None,
            "user_summary_message": "Le module de stratégie interne (Qwen) a rencontré une erreur critique."
        }

# --- Boucle Principale de l'Agent ---
def main_agent_loop():
    rich_print("[bold blue]Assistant de Navigation GUI (Architecture à Deux Niveaux)[/bold blue]")
    logging.info(f"Utilisation VLM Frontend: API Base: {OPENAI_API_BASE_URL}, Modèle: {VLM_MODEL_NAME_FOR_API}")
    logging.info(f"Utilisation LLM Backend (Qwen): API Base: {OPENAI_API_BASE_URL}, Modèle: {QWEN_MODEL_NAME_FOR_API}")
    logging.info(f"Résolution d'écran: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    rich_print(f"Utilisation VLM Frontend: API Base: {OPENAI_API_BASE_URL}, Modèle: {VLM_MODEL_NAME_FOR_API}")
    rich_print(f"Utilisation LLM Backend (Qwen): API Base: {OPENAI_API_BASE_URL}, Modèle: {QWEN_MODEL_NAME_FOR_API}")
    rich_print(f"Résolution d'écran: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    rich_print("Tapez 'exit' ou 'quit' pour quitter.")
    if "internvl3-1b" in VLM_MODEL_NAME_FOR_API or "internvl3-2b" in VLM_MODEL_NAME_FOR_API: # Exemple
        rich_print("[yellow]Note: Les petits modèles VLM peuvent avoir des difficultés avec des prompts JSON complexes.[/yellow]")
    if "4096" in "some_server_config_variable_for_vlm_context_length": # Hypothetical check
        rich_print("[yellow]Attention: Le VLM pourrait être chargé avec une fenêtre de contexte limitée (ex: 4096 tokens). Des prompts longs avec historique peuvent échouer.[/yellow]")


    overall_user_task = ""
    interaction_history = []
    current_task_step_count = 0
    current_vlm_instruction = ""
    consecutive_vlm_failures_for_current_instruction = 0

    while True:
        if not overall_user_task:
            new_task_input = Prompt.ask("\n[bold yellow]Quel est votre objectif global?[/bold yellow]", default="").strip()
            if new_task_input.lower() in ["exit", "quit"]:
                logging.info("Sortie de l'agent."); rich_print("Sortie de l'agent."); break
            if not new_task_input: continue

            overall_user_task = new_task_input
            interaction_history = []
            current_task_step_count = 0
            # L'instruction VLM initiale est l'objectif global de l'utilisateur
            current_vlm_instruction = overall_user_task
            consecutive_vlm_failures_for_current_instruction = 0
            rich_print(f"\n[bold magenta]Nouvel Objectif Global Utilisateur:[/] {overall_user_task}"); play_sound_feedback("ask.wav")

        current_task_step_count += 1
        rich_print(f"\n[bold_white on_blue]>>> Traitement Étape {current_task_step_count}/{MAX_AGENT_STEPS} pour Objectif: '{overall_user_task}' <<[/bold_white on_blue]")
        logging.info(f"Étape {current_task_step_count}/{MAX_AGENT_STEPS} pour '{overall_user_task}'. Instr. VLM: '{current_vlm_instruction}' (Échecs VLM consécutifs sur cette instr.: {consecutive_vlm_failures_for_current_instruction})")
        rich_print(f"Instruction VLM Actuelle: '{current_vlm_instruction}' (Échecs VLM sur cette instr.: {consecutive_vlm_failures_for_current_instruction})")


        if current_task_step_count > MAX_AGENT_STEPS:
            logging.error(f"Nombre maximum d'étapes ({MAX_AGENT_STEPS}) atteint. La tâche '{overall_user_task}' a échoué.")
            rich_print(f"[bold red]Nombre maximum d'étapes ({MAX_AGENT_STEPS}) atteint. La tâche '{overall_user_task}' a échoué.[/bold red]"); play_sound_feedback("error.wav")
            overall_user_task = ""
            continue

        time.sleep(0.3)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        screenshots_folder = "agent_gui_screenshots_api"
        os.makedirs(screenshots_folder, exist_ok=True)
        screenshot_filename = f"etape_{current_task_step_count}_{timestamp}.png"
        screenshot_path = os.path.join(screenshots_folder, screenshot_filename)
        
        screenshot_image_pil = None
        try:
            screenshot_image_pil = ImageGrab.grab()
            screenshot_image_pil.save(screenshot_path)
            logging.info(f"Capture d'écran: {screenshot_path}")
            rich_print(f"Capture d'écran: {screenshot_path}")
        except Exception as e:
            logging.error(f"Erreur lors de la capture d'écran: {e}"); rich_print(f"[red]Erreur capture écran: {e}[/red]"); play_sound_feedback("error.wav")
            time.sleep(1); continue

        image_b64_url_for_vlm = image_to_base64_url(screenshot_image_pil, target_format="PNG") if screenshot_image_pil else None
        if not image_b64_url_for_vlm:
            logging.error("Échec de l'encodage de la capture d'écran pour VLM."); rich_print("[red]Échec encodage capture pour VLM.[/red]"); play_sound_feedback("error.wav")
            time.sleep(1); continue

        vlm_history_summary_for_prompt = []
        for hist_json_str, _ in interaction_history:
            try:
                entry = json.loads(hist_json_str)
                # Rendre le résumé plus concis pour le VLM
                vlm_instr_hist = entry.get("vlm_instruction_given_to_vlm", "N/A")[:100] # Tronquer
                q_dec_obj_hist = entry.get("qwen_decision_obj", {})
                q_dec_type_hist = q_dec_obj_hist.get("decision_type", "N/A")
                q_reason_hist = q_dec_obj_hist.get("reasoning", "N/A")[:100] # Tronquer
                
                summary_line = f"Prev. VLM instruction: '{vlm_instr_hist}...'. Qwen decided: '{q_dec_type_hist}' (Reason: '{q_reason_hist}...') "
                exec_actions_hist = entry.get("executed_action_sequence")
                if exec_actions_hist and isinstance(exec_actions_hist, list):
                    action_types_exec = [a.get('action_type', '?') for a in exec_actions_hist if isinstance(a, dict)]
                    summary_line += f" Executed: {action_types_exec}."
                else:
                    summary_line += " No direct GUI actions taken or other path chosen."
                vlm_history_summary_for_prompt.append(summary_line)
            except Exception as e_hist_parse:
                logging.warning(f"Erreur parsing historique pour prompt VLM: {e_hist_parse}")
                vlm_history_summary_for_prompt.append("(Erreur parsing entrée historique)")
        
        # Stocker l'instruction VLM qui *va être donnée* pour le log, pas celle de la prochaine étape
        vlm_instruction_for_this_turn_log = current_vlm_instruction

        api_messages_for_vlm = build_messages_for_vlm_api(VLM_SYSTEM_PROMPT, current_vlm_instruction, image_b64_url_for_vlm, vlm_history_summary_for_prompt)
        
        vlm_raw_response_str = ""; parsed_vlm_data = None; vlm_api_or_parse_error_msg = None
        try:
            logging.info(f"Envoi de la requête au VLM Frontend (Modèle: {VLM_MODEL_NAME_FOR_API})...")
            rich_print(f"Envoi de la requête au VLM Frontend (Modèle: {VLM_MODEL_NAME_FOR_API})...")
            vlm_completion = client.chat.completions.create(
                model=VLM_MODEL_NAME_FOR_API,
                messages=api_messages_for_vlm,
                max_tokens=1500, # Augmenté légèrement, mais attention au contexte
                temperature=0.01 # Très bas pour la structure JSON
            )
            vlm_raw_response_str = vlm_completion.choices[0].message.content
            logging.debug(f"Réponse Brute VLM (premiers 300): {vlm_raw_response_str[:300]}...")
            rich_print(f"[grey50]Réponse Brute VLM (premiers 300): {vlm_raw_response_str[:300]}...[/grey50]")
            
            parsed_vlm_data = parse_vlm_output_to_sequence(vlm_raw_response_str)
            if parsed_vlm_data is None:
                vlm_api_or_parse_error_msg = "Le parsing du JSON VLM a échoué ou la structure était invalide."
        except Exception as e_vlm_api: # Attraper les erreurs d'API OpenAI/HTTPX aussi
            vlm_api_or_parse_error_msg = f"Erreur API VLM: {str(e_vlm_api)}"
            logging.error(vlm_api_or_parse_error_msg); rich_print(f"[red]{vlm_api_or_parse_error_msg}[/red]");
            # Pas de play_sound_feedback ici, Qwen gère

        current_vlm_status_report_for_qwen = {
            "vlm_output_json_str": vlm_raw_response_str,
            "parsed_vlm_data": parsed_vlm_data,
            "vlm_error_message": vlm_api_or_parse_error_msg,
            "vlm_instruction_given_at_start_of_step": vlm_instruction_for_this_turn_log # Pour le log Qwen
        }

        if vlm_api_or_parse_error_msg or parsed_vlm_data is None:
            consecutive_vlm_failures_for_current_instruction += 1
            logging.warning(f"Échec VLM pour l'instruction actuelle. Total échecs consécutifs pour '{current_vlm_instruction}': {consecutive_vlm_failures_for_current_instruction}")
            rich_print(f"[orange_red1]Échec VLM pour l'instruction actuelle. Total échecs consécutifs pour '{current_vlm_instruction}': {consecutive_vlm_failures_for_current_instruction}[/orange_red1]")
        
        image_b64_url_for_qwen = image_b64_url_for_vlm if "VL" in QWEN_MODEL_NAME_FOR_API.upper() else None
        qwen_decision_obj = get_qwen_strategic_decision(client, overall_user_task, image_b64_url_for_qwen, current_vlm_status_report_for_qwen, interaction_history, consecutive_vlm_failures_for_current_instruction)

        rich_print(f"\n[bold_white on_purple]Décision du Backend Qwen ({qwen_decision_obj.get('decision_type', 'INCONNUE')}):[/]")
        rich_print(f"  [purple]Raisonnement de Qwen:[/purple] {qwen_decision_obj.get('reasoning', 'N/A')}")
        if qwen_decision_obj.get('next_vlm_instruction'):
            rich_print(f"  [purple]Prochaine Instruction VLM (par Qwen):[/purple] {qwen_decision_obj.get('next_vlm_instruction')}")

        actions_to_execute_this_turn = []
        qwen_decision_type = qwen_decision_obj.get("decision_type")
        next_vlm_instr_from_qwen = qwen_decision_obj.get("next_vlm_instruction")
        actions_from_qwen_decision = qwen_decision_obj.get("action_sequence_to_execute")
        
        executed_any_actions_successfully_this_turn = False
        action_execution_failed_mid_sequence = False # Drapeau spécifique pour échec D'UNE action

        if qwen_decision_type == "EXECUTE_VLM_SEQUENCE":
            if parsed_vlm_data and isinstance(parsed_vlm_data.get("action_sequence"), list):
                actions_to_execute_this_turn = parsed_vlm_data["action_sequence"]
                if not actions_to_execute_this_turn: # Si VLM donne une liste vide
                    logging.info("VLM a fourni une action_sequence vide. Aucune action à exécuter de la part du VLM.")
                    rich_print("[grey50]VLM a fourni une action_sequence vide. Aucune action VLM exécutée.[/grey50]")
                consecutive_vlm_failures_for_current_instruction = 0
            else:
                logging.error("Qwen: EXECUTE_VLM_SEQUENCE mais données VLM/action_sequence invalides.")
                rich_print(f"[red]Erreur: Qwen a demandé EXECUTE_VLM_SEQUENCE mais les données VLM sont mauvaises.[/red]")
                if parsed_vlm_data : consecutive_vlm_failures_for_current_instruction +=1
        
        elif qwen_decision_type == "EXECUTE_MODIFIED_SEQUENCE":
            if isinstance(actions_from_qwen_decision, list) and actions_from_qwen_decision:
                actions_to_execute_this_turn = actions_from_qwen_decision
                logging.info(f"Qwen propose une séquence d'actions modifiée/directe de {len(actions_to_execute_this_turn)} action(s).")
                rich_print(f"[green]Exécution de {len(actions_to_execute_this_turn)} micro-actions (dirigées par Qwen)...[/green]")
                if "blind action" in qwen_decision_obj.get("reasoning", "").lower() or \
                   "direct action" in qwen_decision_obj.get("reasoning", "").lower() : # Pour info utilisateur
                    rich_print("[bold orange1]QWEN PREND UNE ACTION À L'AVEUGLE/DIRECTE.[/bold orange1]")
                consecutive_vlm_failures_for_current_instruction = 0
            else:
                logging.error("Qwen: EXECUTE_MODIFIED_SEQUENCE mais 'action_sequence_to_execute' invalide/vide.")
                rich_print(f"[red]Erreur: Qwen a demandé EXECUTE_MODIFIED_SEQUENCE mais sa séquence d'actions est invalide/vide.[/red]")
                consecutive_vlm_failures_for_current_instruction +=1
        
        elif qwen_decision_type == "RETRY_VLM_WITH_NEW_INSTRUCTION":
            if next_vlm_instr_from_qwen and str(next_vlm_instr_from_qwen).strip():
                if current_vlm_instruction != str(next_vlm_instr_from_qwen):
                    consecutive_vlm_failures_for_current_instruction = 0
                current_vlm_instruction = str(next_vlm_instr_from_qwen)
                logging.info(f"Qwen demande un VLM retry. Nouvelle instruction VLM: '{current_vlm_instruction}'")
                rich_print(f"Qwen demande un VLM retry avec nouvelle instruction.")
                play_sound_feedback("ask_2.wav")
            else:
                logging.critical("Qwen: RETRY_VLM sans 'next_vlm_instruction' valide. Passage en TASK_FAILED.")
                rich_print("[red]Critique: Qwen a demandé RETRY_VLM sans instruction valide. Échec de la tâche.[/red]")
                qwen_decision_type = "TASK_FAILED"
                qwen_decision_obj["reasoning"] += " (Erreur interne: next_vlm_instruction manquante pour RETRY)"
        
        elif qwen_decision_type in ["TASK_COMPLETED", "TASK_FAILED"]:
            pass # Géré à la fin de la boucle
        else: # Type de décision Qwen inconnu ou non géré
            logging.error(f"Type de décision Qwen non géré: '{qwen_decision_type}'. Raison Qwen: {qwen_decision_obj.get('reasoning')}")
            rich_print(f"[bold red]Type de décision Qwen non géré: '{qwen_decision_type}'. Traitement comme échec de l'étape.[/bold red]")
            # Conséquence: aucune action, current_vlm_instruction inchangée, le compteur d'échecs VLM pourrait augmenter si VLM avait échoué.
            # Cela pourrait mener à un blocage si Qwen persiste avec un type inconnu.
            # La validation dans get_qwen_strategic_decision devrait empêcher cela.


        if actions_to_execute_this_turn:
            vlm_instruction_marked_finished_in_sequence = False

            for i, micro_action in enumerate(actions_to_execute_this_turn):
                rich_print(f"\n--- Exécution micro-action {i+1}/{len(actions_to_execute_this_turn)} ({'VLM' if qwen_decision_type == 'EXECUTE_VLM_SEQUENCE' else 'Qwen'}) ---")
                m_act_type = micro_action.get("action_type")
                m_desc = micro_action.get("description", "N/A")
                
                action_func_map = {
                    "CLICK": action_click, "DOUBLE_CLICK": action_double_click,
                    "INPUT": action_input_text, "SCROLL": action_scroll,
                    "PRESS_ENTER": action_press_enter, "KEY_PRESS": action_key_press,
                    "PAUSE": action_pause, "FINISHED": action_finished_vlm
                }
                
                success_this_step = False
                if m_act_type in action_func_map:
                    try:
                        if m_act_type in ["CLICK", "DOUBLE_CLICK"]:
                            success_this_step = action_func_map[m_act_type](micro_action["position"], m_desc)
                        elif m_act_type == "INPUT":
                            success_this_step = action_func_map[m_act_type](micro_action["value"], micro_action.get("position"), m_desc)
                        elif m_act_type == "SCROLL":
                            success_this_step = action_func_map[m_act_type](micro_action["direction"], m_desc)
                        elif m_act_type == "KEY_PRESS":
                            success_this_step = action_func_map[m_act_type](micro_action["keys"], m_desc)
                        elif m_act_type == "PAUSE":
                            success_this_step = action_func_map[m_act_type](micro_action["duration_seconds"], m_desc)
                        elif m_act_type == "FINISHED":
                            success_this_step = action_func_map[m_act_type](micro_action.get("reason", "Raison non spécifiée."), m_desc)
                            vlm_instruction_marked_finished_in_sequence = True
                        else: # PRESS_ENTER
                             success_this_step = action_func_map[m_act_type](m_desc)
                    except KeyError as ke:
                        logging.error(f"Champ manquant pour micro-action {m_act_type}: {ke}. Action: {micro_action}")
                        rich_print(f"[red]Champ manquant pour micro-action {m_act_type}: {ke}. Action ignorée.[/red]")
                    except Exception as e_action_exec:
                        logging.error(f"Erreur inattendue durant exécution {m_act_type}: {e_action_exec}")
                        rich_print(f"[red]Erreur exécution {m_act_type}: {e_action_exec}[/red]")
                else:
                    logging.error(f"Type d'action '{m_act_type}' inconnu ou non géré. Action ignorée. Action: {micro_action}")
                    rich_print(f"[red]Type d'action '{m_act_type}' inconnu. Action ignorée.[/red]")
                
                if not success_this_step:
                    logging.error(f"La micro-action {m_act_type} a échoué ou n'a pas pu être exécutée.")
                    rich_print(f"[red]La micro-action {m_act_type} a échoué.[/red]")
                    action_execution_failed_mid_sequence = True
                    break
                
                executed_any_actions_successfully_this_turn = True
                if m_act_type not in ["PAUSE", "FINISHED"]:
                    time.sleep(0.6)
            
            # Mise à jour de l'instruction VLM pour la prochaine itération
            if qwen_decision_type != "RETRY_VLM_WITH_NEW_INSTRUCTION": # Si Qwen n'a pas déjà donné une nouvelle instruction
                if action_execution_failed_mid_sequence:
                    current_vlm_instruction = f"Une action précédente ({m_act_type}) a échoué pour l'objectif '{overall_user_task}'. Évaluez l'écran et proposez une correction ou une nouvelle approche."
                    consecutive_vlm_failures_for_current_instruction = 0 # Reset car c'est un échec d'action, pas de VLM
                elif executed_any_actions_successfully_this_turn:
                    if vlm_instruction_marked_finished_in_sequence:
                         current_vlm_instruction = f"Une action FINISHED a été exécutée pour la sous-tâche de '{overall_user_task}'. Évaluez si l'objectif global est atteint. Si oui, utilisez FINISHED. Sinon, planifiez la prochaine étape."
                    else:
                         current_vlm_instruction = f"Actions (dirigées par {'VLM' if qwen_decision_type == 'EXECUTE_VLM_SEQUENCE' else 'Qwen'}) viennent d'être exécutées pour l'objectif '{overall_user_task}'. Évaluez le nouvel écran. Si l'objectif global n'est pas encore atteint, quel est le prochain sous-objectif VLM ? Si l'objectif global semble atteint, utilisez l'action FINISHED."
                    consecutive_vlm_failures_for_current_instruction = 0
                # Si aucune action n'a été exécutée (ex: liste vide de VLM), current_vlm_instruction ne change pas ici.
                # Qwen devrait donner une nouvelle instruction si VLM donne une liste vide et que ce n'est pas la fin.

        elif qwen_decision_type not in ["TASK_COMPLETED", "TASK_FAILED", "RETRY_VLM_WITH_NEW_INSTRUCTION"]:
             # Si aucune action n'a été exécutée, et que Qwen n'a pas demandé de retry VLM,
             # la current_vlm_instruction reste la même. Le compteur d'échec VLM (si VLM a échoué)
             # est déjà incrémenté.
             logging.info("Aucune action exécutée ce tour. L'instruction VLM reste la même ou a été modifiée par Qwen (RETRY).")


        # Enregistrement de l'historique
        current_history_entry_data_for_prompt = {
            "step_count": current_task_step_count,
            "vlm_instruction_given_to_vlm": vlm_instruction_for_this_turn_log, # Instruction donnée au VLM à cette étape
            "qwen_decision_obj": { # Résumé simplifié pour le prompt
                "decision_type": qwen_decision_obj.get("decision_type"),
                "reasoning": qwen_decision_obj.get("reasoning", "N/A")[:150] + "..." if len(qwen_decision_obj.get("reasoning", "")) > 150 else qwen_decision_obj.get("reasoning", "N/A"),
            },
            "executed_action_sequence": actions_to_execute_this_turn if actions_to_execute_this_turn else None
        }
        history_log_details_for_file = ( # Log plus détaillé pour débogage
            f"--- Étape {current_task_step_count} pour Objectif: '{overall_user_task}' ---\n"
            f"Instruction VLM donnée à cette étape: {vlm_instruction_for_this_turn_log}\n"
            f"Réponse Brute VLM:\n{vlm_raw_response_str}\n"
            f"JSON Parsé VLM (si succès):\n{json.dumps(parsed_vlm_data, indent=2, ensure_ascii=False) if parsed_vlm_data else 'N/A ou Échec Parsing'}\n"
            f"Erreur VLM (si applicable): {vlm_api_or_parse_error_msg if vlm_api_or_parse_error_msg else 'Aucune'}\n\n"
            f"Réponse Brute Qwen:\n{qwen_decision_obj.get('raw_qwen_response_str_for_debug', 'Non disponible')}\n"
            f"Décision Qwen (JSON Complet):\n{json.dumps(qwen_decision_obj, indent=2, ensure_ascii=False)}\n"
            f"Actions Exécutées (si applicable):\n{json.dumps(actions_to_execute_this_turn, indent=2, ensure_ascii=False) if actions_to_execute_this_turn else 'Aucune'}\n"
            f"Prochaine Instruction VLM (si définie par Qwen ou logique interne): {current_vlm_instruction}\n"
            f"-----------------------------------\n"
        )
        interaction_history.append((json.dumps(current_history_entry_data_for_prompt, ensure_ascii=False), history_log_details_for_file))
        # Optionnel: écrire history_log_details_for_file dans un fichier log séparé
        with open(os.path.join(screenshots_folder, "detailed_interaction_log.txt"), "a", encoding="utf-8") as log_file:
            log_file.write(history_log_details_for_file)


        if qwen_decision_type == "TASK_COMPLETED":
            msg = qwen_decision_obj.get('user_summary_message', 'Tâche terminée avec succès!')
            rich_print(f"\n[bold_green on_black]QWEN: OBJECTIF TERMINÉ! 🎉 {msg}[/]"); play_sound_feedback("task_completed.wav")
            overall_user_task = ""
        elif qwen_decision_type == "TASK_FAILED":
            msg = qwen_decision_obj.get('user_summary_message', 'Échec de la tâche.')
            rich_print(f"\n[bold_red on_black]QWEN: OBJECTIF ÉCHOUÉ. 😥 {msg}[/]"); play_sound_feedback("error.wav")
            overall_user_task = ""

        if not overall_user_task:
            logging.info("--- Réinitialisation pour un nouvel objectif utilisateur global ---")
            rich_print("--- Réinitialisation pour un nouvel objectif utilisateur global ---")


if __name__ == "__main__":
    if AUDIO_ENABLED:
        audio_dir = "audio_feedback"
        os.makedirs(audio_dir, exist_ok=True)
        sr = 16000
        dur = 0.15

        try:
            t_np = np.linspace(0, dur, int(sr * dur), False)
            sound_configs = [
                ("ask.wav", 660), ("ok.wav", 880), ("ask_2.wav", 550),
                ("error.wav", 330), ("task_completed.wav", 1046)
            ]
            dummy_sound_data_fallback = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(sr * 0.1), False))

            for fname, freq in sound_configs:
                f_path = os.path.join(audio_dir, fname)
                if not os.path.exists(f_path):
                    try:
                        sound_data = 0.3 * np.sin(2 * np.pi * freq * t_np)
                        fade_len = int(sr * 0.05)
                        if len(sound_data) > fade_len:
                             sound_data[-fade_len:] *= np.linspace(1, 0, fade_len)
                        sf.write(f_path, sound_data, sr, format='WAV', subtype='PCM_16')
                    except Exception as e_sf_write:
                        logging.warning(f"Impossible d'écrire {fname} ({e_sf_write}), fallback.")
                        try: sf.write(f_path, dummy_sound_data_fallback, sr, format='WAV', subtype='PCM_16')
                        except Exception as e_sf_fallback:
                            logging.error(f"Échec écriture fallback {fname}: {e_sf_fallback}")
                            AUDIO_ENABLED = False; break
        except NameError:
            logging.warning("Numpy non trouvé. Pas de génération de sons. (pip install numpy)")
            AUDIO_ENABLED = False
        except Exception as e_audio_create:
            logging.warning(f"Erreur création sons: {e_audio_create}")
            AUDIO_ENABLED = False
            
    try:
        main_agent_loop()
    except KeyboardInterrupt:
        rich_print("\n[bold yellow]Interruption utilisateur. Sortie de l'agent.[/bold yellow]")
        logging.info("Agent interrompu par l'utilisateur.")
    except Exception as e_main:
        logging.critical(f"Erreur non gérée dans la boucle principale: {e_main}", exc_info=True)
        rich_print(f"[bold red]Erreur critique non gérée dans la boucle principale: {e_main}[/bold red]")
    finally:
        logging.info("Arrêt de l'agent.")
        rich_print("Agent arrêté.")
