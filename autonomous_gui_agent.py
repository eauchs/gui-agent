import json
import math
import os
import time
import tkinter as tk
from tkinter import Label
import pyautogui
from PIL import Image, ImageGrab # Pillow
from pynput.mouse import Controller as MouseController
from rich import print
from rich.prompt import Prompt
import base64
import io
import re
import unicodedata

try:
    from openai import OpenAI
except ImportError:
    print("[bold red]Erreur Fatale: Librairie OpenAI non trouvée. Veuillez l'installer.[/bold red]")
    print("Exécutez: pip install openai")
    exit()

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_ENABLED = True
except ImportError:
    print("[yellow]Attention: sounddevice ou soundfile non trouvé. Le retour audio sera désactivé.[/yellow]")
    AUDIO_ENABLED = False

try:
    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
except Exception as e:
    print(f"[red]Erreur lors de la récupération de la taille de l'écran via pyautogui: {e}[/red]")
    print("[yellow]Utilisation par défaut de 1920x1080. Ajustez si nécessaire.[/yellow]")
    SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

# --- Configuration des Constantes ---
OVERLAY_DURATION = 0.8 # Durée d'affichage des overlays d'action
HIGHLIGHT_DURATION = 0.15 # Durée de l'animation de surbrillance du clic
CURSOR_ANIMATION_DURATION = 0.2 # Durée de l'animation de mouvement du curseur
mouse_controller = MouseController()

OPENAI_API_BASE_URL = "http://localhost:1234/v1" # Endpoint de votre serveur (ex: LM Studio)
OPENAI_API_KEY = "lm-studio" # Clé API si nécessaire pour votre serveur

# --- NOMS DES MODÈLES ---
# REMPLACEZ CES VALEURS PAR LES NOMS EXACTS DE VOS MODÈLES TELS QUE CONFIGURÉS DANS VOTRE SERVEUR API
VLM_MODEL_NAME_FOR_API = "internvl3-2b-instruct" # Modèle VLM Frontend (perception)
QWEN_MODEL_NAME_FOR_API = "josiefied-qwen3-8b-abliterated-v1"   # Modèle LLM Backend (raisonnement stratégique)

# Seuil avant que Qwen puisse tenter une action "à l'aveugle"
MAX_CONSECUTIVE_VLM_FAILURES_BEFORE_BLIND_ACTION = 2

# Initialisation du client OpenAI (utilisé pour les deux modèles via le même endpoint)
try:
    client = OpenAI(base_url=OPENAI_API_BASE_URL, api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"[bold red]Erreur Fatale: Impossible d'initialiser le client OpenAI: {e}[/bold red]")
    exit()

# --- Prompt Système pour le VLM Frontend (Perception) ---
VLM_SYSTEM_PROMPT = f"""
You are a VLM assistant for a macOS GUI agent. You analyze screenshots and follow SPECIFIC INSTRUCTIONS from a supervisor LLM.
Your screen resolution is {SCREEN_WIDTH}x{SCREEN_HEIGHT}.
YOUR ENTIRE RESPONSE MUST BE A SINGLE, VALID JSON OBJECT. ONLY THE JSON.

JSON structure: {{ "global_thought": {{...}}, "action_sequence": [...] }}

"global_thought" keys (try your best to provide all, if a key is truly not applicable or unknown based on the screen, use "N/A"):
1.  "Current State Summary": Your summary of task progress for the current instruction.
2.  "User's Current Instruction": Restate the specific instruction you are working on.
3.  "Previous Action Assessment": Outcome of YOUR VLM's PREVIOUS action_sequence for THIS instruction (e.g., "Successfully opened Spotlight", "Clicked button, new dialog appeared", "N/A if first VLM step for this instruction").
4.  "Current Screen Analysis (Brief)": Describe the screenshot relevant to the CURRENT instruction.
5.  "Next Immediate Sub-goal for THIS Instruction": Your specific sub-goal for THIS instruction.
6.  "Action Justification & Selection": Justify your proposed 'action_sequence' for THIS sub-goal.
7.  "Anticipated Next Step AFTER THIS sequence": What you expect on screen after your actions, related to THIS instruction.

"action_sequence": A LIST of micro-action objects. Available "action_type":
"CLICK" ({{ "position": [norm_x, norm_y], "description": "..." }}),
"DOUBLE_CLICK" ({{ "position": [norm_x, norm_y], "description": "..." }}),
"INPUT" ({{ "value": "text", "position": [opt_norm_x, opt_norm_y], "description": "..." }}),
"SCROLL" ({{ "direction": "up/down", "description": "..." }}),
"PRESS_ENTER" ({{ "description": "..." }}),
"KEY_PRESS" ({{ "keys": ["MOD", "KEY"], "description": "..." }}), Example: ["COMMAND", "SPACE"],
"PAUSE" ({{ "duration_seconds": float, "description": "..." }}),
"FINISHED" ({{ "reason": "...", "description": "Instruction complete." }}) -> Use this if your CURRENT INSTRUCTION is fully completed.

Focus ONLY on the immediate instruction. The supervisor LLM (Qwen) handles the overall user goal.
If the instruction is to "confirm if X is visible", your action_sequence might be empty, and "global_thought" confirms it. Then use "FINISHED".
"""

# --- Fonctions Audio ---
def play_sound_feedback(sound_file_name):
    if not AUDIO_ENABLED:
        return
    try:
        audio_dir = "audio_feedback"
        os.makedirs(audio_dir, exist_ok=True) # Crée le dossier s'il n'existe pas
        file_path = os.path.join(audio_dir, sound_file_name)
        if os.path.exists(file_path):
            data, fs = sf.read(file_path, dtype='float32')
            sd.play(data, fs)
            # sd.wait() # Décommentez si vous voulez bloquer l'exécution jusqu'à la fin du son
        else:
            print(f"[yellow]Fichier audio non trouvé: {file_path}[/yellow]")
    except Exception as e:
        print(f"[yellow]Impossible de jouer le son {sound_file_name}: {e}[/yellow]")

# --- Fonctions Utilitaires pour l'Overlay et l'Animation du Curseur ---
def animate_cursor_movement(start_x, start_y, end_x, end_y, duration=CURSOR_ANIMATION_DURATION):
    steps = max(1, int(duration * 100)) # S'assurer qu'il y a au moins 1 étape
    for i in range(steps + 1):
        t = i / steps
        x = start_x + (end_x - start_x) * t
        y = start_y + (end_y - start_y) * t
        try:
            mouse_controller.position = (x, y)
        except: # Fallback si pynput échoue (rare)
            pyautogui.moveTo(x,y, duration=duration/steps/2) # Diviser la durée pour chaque micro-mouvement
        time.sleep(duration / steps)

def highlight_click_position(x, y, duration=HIGHLIGHT_DURATION):
    current_pos = pyautogui.position()
    animate_cursor_movement(current_pos[0], current_pos[1], x, y) # Déplace le curseur vers la cible
    
    # Petite animation circulaire autour du point de clic
    radius = 7
    steps = 8
    for i in range(steps + 1):
        angle = 2 * math.pi * i / steps
        new_x = x + radius * math.cos(angle)
        new_y = y + radius * math.sin(angle)
        try:
            mouse_controller.position = (new_x, new_y)
        except:
            pyautogui.moveTo(new_x, new_y, duration=duration/(steps+1)/2)
        time.sleep(duration / (steps+1)) # Diviser la durée totale de l'highlight
    try: # Ramener le curseur au centre exact
        mouse_controller.position = (x, y)
    except:
        pyautogui.moveTo(x,y)

def create_action_overlay(action_text, x, y, color="lime", duration=OVERLAY_DURATION):
    try:
        root = tk.Tk()
        root.overrideredirect(True) # Fenêtre sans bordures ni barre de titre
        root.attributes("-topmost", True) # Toujours au-dessus
        root.attributes("-alpha", 0.75) # Transparence
        root.config(bg="black") # Couleur de fond de la fenêtre (pas du label)

        lines = action_text.split('\n')
        max_line_len = max(len(line) for line in lines) if lines else 0
        
        # Estimation de la taille basée sur la police et le contenu
        approx_char_width = 9
        approx_line_height = 22
        padding_x = 20
        padding_y = 10

        overlay_width = (max_line_len * approx_char_width) + padding_x
        overlay_height = (len(lines) * approx_line_height) + padding_y
        
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        
        # Positionner l'overlay avec un décalage, en s'assurant qu'il reste dans l'écran
        offset_x, offset_y = 25, 25
        final_x = min(max(0, x + offset_x), screen_w - overlay_width)
        final_y = min(max(0, y + offset_y), screen_h - overlay_height)

        root.geometry(f"{int(overlay_width)}x{int(overlay_height)}+{int(final_x)}+{int(final_y)}")
        
        label = Label(root, text=action_text, fg=color, bg="black", font=("Arial", 12, "bold"), justify=tk.LEFT)
        label.pack(padx=padding_x//2, pady=padding_y//2, expand=True, fill='both')
        
        # Utiliser une lambda pour s'assurer que winfo_exists est appelé au moment de l'exécution de after
        root.after(int(duration * 1000), lambda: root.destroy() if root.winfo_exists() else None)
        root.update() # Forcer la mise à jour de la fenêtre
        return root
    except Exception as e:
        print(f"[yellow]Avertissement Overlay: {e}. Les actions GUI continueront.[/yellow]")
        return None

def safe_destroy_overlay(overlay_ref):
    if overlay_ref and overlay_ref.winfo_exists(): # Vérifier si la référence existe et si la fenêtre existe
        overlay_ref.destroy()

# --- Fonctions d'Action GUI (Ex: action_click, action_input_text, etc.) ---
def action_click(position_norm, description=""):
    x, y = round(position_norm[0] * SCREEN_WIDTH), round(position_norm[1] * SCREEN_HEIGHT)
    print(f"Exécution: CLICK à ({x}, {y}) Description: {description}")
    overlay = create_action_overlay(f"CLICK\n({x},{y})\n{description[:30]}", x, y)
    highlight_click_position(x, y)
    pyautogui.click(x=x, y=y)
    if overlay: overlay.after(200, lambda: safe_destroy_overlay(overlay)) # Fermer un peu après

def action_double_click(position_norm, description=""):
    x, y = round(position_norm[0] * SCREEN_WIDTH), round(position_norm[1] * SCREEN_HEIGHT)
    print(f"Exécution: DOUBLE_CLICK à ({x}, {y}) Description: {description}")
    overlay = create_action_overlay(f"DBL_CLICK\n({x},{y})\n{description[:30]}", x, y)
    highlight_click_position(x, y)
    pyautogui.doubleClick(x=x, y=y, interval=0.1) # Intervalle court pour double clic rapide
    if overlay: overlay.after(200, lambda: safe_destroy_overlay(overlay))

def action_input_text(value_to_type, position_norm=None, description=""):
    print(f"Exécution: INPUT '{value_to_type}' Description: {description}")
    overlay_x, overlay_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 # Position par défaut de l'overlay
    if position_norm: # Si une position est fournie, cliquer d'abord là
        x, y = round(position_norm[0] * SCREEN_WIDTH), round(position_norm[1] * SCREEN_HEIGHT)
        print(f"Clic à ({x}, {y}) avant de taper.")
        highlight_click_position(x, y); pyautogui.click(x=x, y=y); time.sleep(0.2) # Petite pause après le clic
        overlay_x, overlay_y = x, y # Mettre l'overlay près du clic
    
    overlay_text = f"TYPE:\n{value_to_type[:25]}{'...' if len(value_to_type) > 25 else ''}\n{description[:30]}"
    overlay = create_action_overlay(overlay_text, overlay_x, overlay_y)
    pyautogui.write(value_to_type, interval=0.03) # Intervalle entre les touches pour plus de naturel
    if overlay: overlay.after(200, lambda: safe_destroy_overlay(overlay))

def action_scroll(direction, description=""):
    scroll_clicks = 10 # Nombre d'unités de défilement
    amount = -scroll_clicks if direction.lower() == "up" else scroll_clicks
    print(f"Exécution: SCROLL {direction.upper()} Description: {description}")
    overlay = create_action_overlay(f"SCROLL {direction.upper()}\n{description[:30]}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    pyautogui.scroll(amount)
    if overlay: overlay.after(200, lambda: safe_destroy_overlay(overlay))

def action_press_enter(description=""):
    print(f"Exécution: PRESS_ENTER Description: {description}")
    overlay = create_action_overlay(f"ENTER\n{description[:30]}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    pyautogui.press("enter")
    if overlay: overlay.after(200, lambda: safe_destroy_overlay(overlay))

def action_key_press(keys_to_press_original, description=""):
    print(f"Description: {description}")
    processed_keys_list = []
    for key_item in keys_to_press_original:
        if isinstance(key_item, str) and " " in key_item:
            parts = key_item.lower().split()
            processed_keys_list.extend(parts) if len(parts) > 1 else processed_keys_list.append(key_item.lower())
        elif isinstance(key_item, str):
            processed_keys_list.append(key_item.lower())
    
    if not processed_keys_list:
        print(f"[red]Erreur KEY_PRESS: Aucune touche valide après traitement. Original: {keys_to_press_original}[/red]")
        play_sound_feedback("error.wav"); return False

    pyautogui_keys = []
    original_keys_for_display = list(keys_to_press_original) # Pour l'affichage
    for key_name_lower in processed_keys_list:
        # Mappages de base pour pyautogui
        if key_name_lower in ["command", "cmd", "win", "super"]: pyautogui_keys.append("command") # 'win' ou 'super' pour compatibilité
        elif key_name_lower in ["option", "alt"]: pyautogui_keys.append("option")
        elif key_name_lower in ["control", "ctrl"]: pyautogui_keys.append("ctrl")
        elif key_name_lower == "shift": pyautogui_keys.append("shift")
        elif key_name_lower == "space": pyautogui_keys.append("space")
        elif key_name_lower == "delete": pyautogui_keys.append("backspace") # 'delete' est souvent 'backspace'
        elif key_name_lower in ["enter", "return"]: pyautogui_keys.append("enter")
        elif key_name_lower == "tab": pyautogui_keys.append("tab")
        elif key_name_lower in ["escape", "esc"]: pyautogui_keys.append("esc")
        elif key_name_lower == "up_arrow": pyautogui_keys.append("up")
        elif key_name_lower == "down_arrow": pyautogui_keys.append("down")
        elif key_name_lower == "left_arrow": pyautogui_keys.append("left")
        elif key_name_lower == "right_arrow": pyautogui_keys.append("right")
        elif re.match(r"f\d{1,2}", key_name_lower): pyautogui_keys.append(key_name_lower) # F1-F12
        elif len(key_name_lower) == 1 and (key_name_lower.isalnum() or key_name_lower in [',', '.', '/', ';', "'", '[', ']', '\\', '-', '=', '`']):
            pyautogui_keys.append(key_name_lower) # Caractères alphanumériques et symboles courants
        else:
            print(f"[red]Attention: Touche inconnue '{key_name_lower}' dans KEY_PRESS. Ignorée. Original: {original_keys_for_display}[/red]")
            continue
            
    if not pyautogui_keys:
        print(f"[red]Erreur KEY_PRESS: Aucune touche PyAutoGUI valide à presser. Original VLM: {original_keys_for_display}[/red]")
        play_sound_feedback("error.wav"); return False
    
    print(f"Exécution: KEY_PRESS PyAutoGUI: {pyautogui_keys} (Original VLM: {original_keys_for_display})")
    overlay = create_action_overlay(f"KEY_PRESS:\n{', '.join(original_keys_for_display)}\n{description[:30]}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, color="orange")
    
    try:
        # Logique améliorée pour hotkey vs pressions séquentielles
        potential_modifiers = ["command", "option", "ctrl", "shift"]
        # Vérifier si c'est un raccourci standard (modificateurs suivis d'une touche non-modificateur)
        is_standard_hotkey = False
        if len(pyautogui_keys) > 1:
            last_key = pyautogui_keys[-1]
            first_keys = pyautogui_keys[:-1]
            if last_key not in potential_modifiers and all(k in potential_modifiers for k in first_keys):
                is_standard_hotkey = True
        
        if is_standard_hotkey:
            pyautogui.hotkey(*pyautogui_keys)
        elif len(pyautogui_keys) == 1: # Une seule touche à presser
            pyautogui.press(pyautogui_keys[0])
        else: # Séquence plus complexe ou multiples touches non-modificatrices
            print(f"[yellow]Séquence de touches complexe, tentative de pressions individuelles: {pyautogui_keys}[/yellow]")
            for key_to_press in pyautogui_keys:
                pyautogui.press(key_to_press)
                time.sleep(0.05) # Petite pause entre les pressions
        
        if overlay: overlay.after(200, lambda: safe_destroy_overlay(overlay)); return True
    except Exception as e:
        print(f"[bold red]Erreur durant KEY_PRESS: {e}[/bold red]"); play_sound_feedback("error.wav")
        if overlay: overlay.after(200, lambda: safe_destroy_overlay(overlay)); return False

def action_pause(duration_seconds, description=""):
    print(f"Exécution: PAUSE pour {duration_seconds}s Description: {description}")
    overlay = create_action_overlay(f"PAUSE {duration_seconds}s\n{description[:30]}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, color="blue")
    time.sleep(float(duration_seconds))
    if overlay: safe_destroy_overlay(overlay) # Détruire immédiatement après la pause

def action_finished_vlm(reason="VLM: L'instruction semble terminée.", description=""):
    print(f"Action VLM: FINISHED. Raison VLM: {reason} Description: {description}")
    play_sound_feedback("ok.wav") # Son spécifique pour le 'finished' du VLM

# --- Fonctions de l'Agent (image_to_base64_url, parse_vlm_output_to_sequence) ---
def image_to_base64_url(image_path, format="PNG"):
    try:
        with Image.open(image_path) as img:
            if format.upper() == "JPEG" and (img.mode == 'RGBA' or img.mode == 'LA' or img.mode == 'P'):
                img = img.convert('RGB')
            # PNG peut gérer la transparence, pas besoin de conversion agressive sauf si mode non supporté
            buffered = io.BytesIO()
            img.save(buffered, format=format)
            return f"data:image/{format.lower()};base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    except Exception as e:
        print(f"[red]Erreur d'encodage de l'image {image_path}: {e}[/red]")
        return None

def parse_vlm_output_to_sequence(vlm_response_str: str):
    json_str_to_parse = None
    cleaned_str = vlm_response_str.strip()
    
    # Essayer d'extraire le JSON d'un bloc Markdown ```json ... ```
    match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", cleaned_str, re.DOTALL)
    if match:
        json_str_to_parse = match.group(1)
        print("[grey50]JSON extrait d'un bloc Markdown VLM.[/grey50]")
    else:
        # Sinon, essayer de trouver le premier '{' et le dernier '}'
        first_brace = cleaned_str.find('{')
        last_brace = cleaned_str.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            json_str_to_parse = cleaned_str[first_brace : last_brace + 1]
            print("[grey50]JSON extrait de VLM par première/dernière accolade.[/grey50]")
        else:
            print(f"[red]Erreur VLM: Aucune structure JSON trouvée dans la réponse VLM: {cleaned_str[:200]}...[/red]")
            return None # Pas de JSON à parser

    try:
        # Nettoyer les caractères de contrôle non standard avant le parsing (sauf tab, newline, carriage return)
        cleaned_for_parsing = "".join(ch for ch in json_str_to_parse if unicodedata.category(ch)[0] != "C" or ch in ('\t', '\n', '\r'))
        decoded_data = json.loads(cleaned_for_parsing)

        if not isinstance(decoded_data, dict):
            print(f"[red]Erreur VLM: Le JSON décodé n'est pas un dictionnaire. Type: {type(decoded_data)}[/red]")
            return None
        
        global_thought_raw = decoded_data.get("global_thought")
        action_sequence_raw = decoded_data.get("action_sequence")

        if global_thought_raw is None:
            print("[red]Erreur VLM: Clé 'global_thought' manquante.[/red]")
            return None # 'global_thought' est essentiel
        if not isinstance(global_thought_raw, dict):
            print("[red]Erreur VLM: 'global_thought' n'est pas un objet.[/red]")
            return None

        # --- Assouplissement du parsing pour les sous-clés de global_thought ---
        parsed_global_thought = {}
        # Clés attendues basées sur VLM_SYSTEM_PROMPT (plus robuste que l'extraction par regex)
        expected_thought_keys = [
            "Current State Summary", "User's Current Instruction", "Previous Action Assessment",
            "Current Screen Analysis (Brief)", "Next Immediate Sub-goal for THIS Instruction",
            "Action Justification & Selection", "Anticipated Next Step AFTER THIS sequence"
        ]
        incomplete_thought = False
        for key in expected_thought_keys:
            if key in global_thought_raw:
                parsed_global_thought[key] = global_thought_raw[key]
            else:
                # Si une clé manque, on la loggue mais on continue avec une valeur par défaut
                print(f"[yellow]Avertissement VLM: 'global_thought' manque la clé '{key}'. Utilisation de 'N/A'.[/yellow]")
                parsed_global_thought[key] = "N/A (non fourni par VLM)"
                incomplete_thought = True
        
        if incomplete_thought:
             print("[grey50]Le 'global_thought' du VLM était incomplet mais a été traité avec des valeurs par défaut.[/grey50]")
        # --- Fin de l'assouplissement ---

        if action_sequence_raw is None:
            print("[red]Erreur VLM: Clé 'action_sequence' manquante.[/red]")
            # On pourrait retourner ici, mais si Qwen peut agir à l'aveugle,
            # il est peut-être utile de lui donner la pensée (même partielle) du VLM.
            # Pour l'instant, on exige une action_sequence (même vide si c'est un FINISHED).
            return None
        if not isinstance(action_sequence_raw, list):
            print("[red]Erreur VLM: 'action_sequence' n'est pas une liste.[/red]")
            return None
        
        validated_sequence = []
        for i, micro_action in enumerate(action_sequence_raw):
            if not isinstance(micro_action, dict):
                print(f"[red]Erreur VLM: Micro-action {i} n'est pas un objet.[/red]"); return None
            
            action_type = micro_action.get("action_type")
            description = micro_action.get("description", f"Étape VLM {i+1}") # Description par défaut
            micro_action["description"] = description # S'assurer qu'elle est là

            if not action_type:
                print(f"[red]Erreur VLM: Micro-action {i} manque 'action_type'.[/red]"); return None
            
            valid_action_types = ["CLICK", "DOUBLE_CLICK", "INPUT", "SCROLL", "PRESS_ENTER", "KEY_PRESS", "PAUSE", "FINISHED"]
            if action_type not in valid_action_types:
                print(f"[red]Erreur VLM: Micro-action {i} a un 'action_type' inconnu: {action_type}[/red]"); return None

            # Validations spécifiques par type d'action (champs requis)
            if action_type in ["CLICK", "DOUBLE_CLICK"] and "position" not in micro_action:
                print(f"[red]Erreur VLM: Micro-action {i} ({action_type}) manque 'position'.[/red]"); return None
            if action_type == "INPUT" and "value" not in micro_action: # 'position' est optionnelle pour INPUT
                print(f"[red]Erreur VLM: Micro-action {i} (INPUT) manque 'value'.[/red]"); return None
            if action_type == "SCROLL" and "direction" not in micro_action:
                print(f"[red]Erreur VLM: Micro-action {i} (SCROLL) manque 'direction'.[/red]"); return None
            if action_type == "KEY_PRESS" and "keys" not in micro_action:
                print(f"[red]Erreur VLM: Micro-action {i} (KEY_PRESS) manque 'keys'.[/red]"); return None
            if action_type == "PAUSE" and "duration_seconds" not in micro_action:
                print(f"[red]Erreur VLM: Micro-action {i} (PAUSE) manque 'duration_seconds'.[/red]"); return None
            if action_type == "FINISHED" and "reason" not in micro_action:
                micro_action["reason"] = "Le VLM a déterminé que l'instruction est terminée." # Raison par défaut

            # Validation et normalisation de 'position'
            if "position" in micro_action and micro_action["position"] is not None:
                pos = micro_action["position"]
                if not (isinstance(pos, list) and len(pos) == 2 and all(isinstance(p, (float, int, str)) for p in pos)):
                    print(f"[red]Erreur VLM: Format de 'position' invalide pour micro_action {i}: {pos}[/red]"); return None
                try:
                    raw_x, raw_y = float(pos[0]), float(pos[1])
                    # Normaliser si les coordonnées sont en pixels absolus (supérieurs à 1.0 ou entiers > 1)
                    norm_x = raw_x / SCREEN_WIDTH if raw_x > 1.0 or (isinstance(pos[0], int) and abs(pos[0]) > 1) else raw_x
                    norm_y = raw_y / SCREEN_HEIGHT if raw_y > 1.0 or (isinstance(pos[1], int) and abs(pos[1]) > 1) else raw_y
                    micro_action["position"] = [max(0.0, min(1.0, norm_x)), max(0.0, min(1.0, norm_y))] # Clamper entre 0 et 1
                except ValueError:
                    print(f"[red]Erreur VLM: Impossible de convertir la position pour micro_action {i}: {pos}[/red]"); return None
            
            validated_sequence.append(micro_action)
        
        # Si tout est bon, retourner la pensée parsée (potentiellement avec N/A) et la séquence validée
        return {"global_thought": parsed_global_thought, "action_sequence": validated_sequence}

    except json.JSONDecodeError as e:
        print(f"[red]Erreur VLM JSONDecodeError: {e}. Tentative sur: '{cleaned_for_parsing[:300]}...'[/red]")
        return None
    except Exception as e:
        print(f"[red]Erreur VLM inattendue lors du parsing: {e}. Entrée: '{cleaned_str[:200]}...'[/red]")
        return None

# --- Construction de Messages pour le VLM Frontend ---
def build_messages_for_vlm_api(system_prompt_content, current_instruction_for_vlm, image_base64_url, vlm_execution_history_summary_list):
    # current_instruction_for_vlm: L'instruction spécifique que Qwen donne au VLM.
    # vlm_execution_history_summary_list: Liste de chaînes résumant les actions VLM précédentes et leurs résultats (du point de vue de Qwen).
    
    user_text_content = f"Current Specific Instruction from Supervisor: {current_instruction_for_vlm}"

    if vlm_execution_history_summary_list:
        user_text_content += "\n\n--- Summary of Your Previous VLM Steps & Outcomes (for this overall task, max 3 shown) ---"
        for i, summary_line in enumerate(vlm_execution_history_summary_list[-3:]): # Montrer les 3 dernières étapes résumées
            user_text_content += f"\n{i+1}. {summary_line}"
    else:
        user_text_content += "\n\n--- No previous VLM steps for this current VLM instruction phase. ---"

    user_text_content += "\n\n--- Current Situation ---"
    user_text_content += "\nAnalyze the current screen and follow the 'Current Specific Instruction from Supervisor'."
    user_text_content += "\nRemember, 'Command+Space' TOGGLES Spotlight. Use PAUSES effectively."
    user_text_content += "\nIf the instruction is completed, use the 'FINISHED' action."

    messages = [{"role": "system", "content": system_prompt_content}]
    content_list = [{"type": "text", "text": user_text_content}]
    if image_base64_url:
        content_list.append({"type": "image_url", "image_url": {"url": image_base64_url, "detail": "high"}}) # "high" pour GUI
    messages.append({"role": "user", "content": content_list})
    return messages

# --- Fonction pour Interroger Qwen (LLM Backend) ---
def get_qwen_strategic_decision(api_client, overall_user_goal, image_base64_url, # Peut être None si Qwen n'est pas multimodal
                                current_vlm_status_report, # Dictionnaire: {"vlm_output_json_str": str, "parsed_vlm_data": dict|None, "vlm_error_message": str|None}
                                full_interaction_history, # Liste de (enriched_history_entry_json_str, raw_responses_str)
                                consecutive_vlm_failures_for_current_instruction_count):
    
    qwen_system_prompt = f"""
You are an expert strategic supervisor for a macOS GUI automation agent.
The agent has a VLM frontend that analyzes screenshots + specific instructions you provide, and proposes GUI 'action_sequence' in JSON.
You (Qwen) are the brain: receive 'overall_user_goal', give specific instructions to VLM, analyze VLM's JSON output, and decide the final action_sequence or next VLM instruction.
Screen resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}. VLM actions: CLICK, INPUT, KEY_PRESS, FINISHED, etc.

You will receive:
- 'overall_user_goal'.
- 'current_vlm_status_report': A dict with 'vlm_output_json_str' (VLM's raw JSON attempt), 'parsed_vlm_data' (parsed VLM output, or null if parse failed), and 'vlm_error_message' (if VLM failed badly).
- 'image_base64_url': Current screenshot (ONLY if you are multimodal Qwen-VL, otherwise ignore).
- 'interaction_history_summary': Summary of VLM outputs & your past Qwen decisions.
- 'consecutive_vlm_failures_for_current_instruction_count': How many times VLM has failed (bad JSON or supervisor rejection) for the CURRENT specific VLM instruction.

YOUR TASKS:
1.  CRITICALLY EVALUATE VLM's output (if 'parsed_vlm_data' is available) or its failure.
    - Is VLM's 'global_thought.Previous Action Assessment' correct? (Often it's wrong or "N/A").
    - Does VLM's plan help the 'overall_user_goal'? Is it stuck/looping?
    - If VLM used "FINISHED", is its specific instruction truly complete for the overall goal?
2.  DECIDE the next course of action.

**BLIND ACTION STRATEGY:** If 'consecutive_vlm_failures_for_current_instruction_count' is >= {MAX_CONSECUTIVE_VLM_FAILURES_BEFORE_BLIND_ACTION} OR if 'parsed_vlm_data' is consistently null/unusable, you MAY attempt a "blind action".
This means proposing an 'action_sequence_to_execute' based on the 'overall_user_goal' and general macOS knowledge, even if VLM provides no usable visual input for this step.
Clearly state in 'reasoning' if you are taking a blind action (e.g., "VLM failed {MAX_CONSECUTIVE_VLM_FAILURES_BEFORE_BLIND_ACTION} times, attempting blind action: open Spotlight and type X.").
Use with caution. Prefer giving VLM simpler, more direct instructions first. If taking blind action, the 'action_sequence_to_execute' should be self-contained and robust.

YOUR RESPONSE MUST BE A SINGLE VALID JSON OBJECT:
{{
  "decision_type": "[EXECUTE_VLM_SEQUENCE | EXECUTE_MODIFIED_SEQUENCE | RETRY_VLM_WITH_NEW_INSTRUCTION | TASK_COMPLETED | TASK_FAILED]",
  "reasoning": "Your concise explanation. State if taking BLIND action and why.",
  "action_sequence_to_execute": [array of VLM micro-actions, or null],  // Required for EXECUTE_*
  "next_vlm_instruction": "New specific instruction for VLM, or null", // Required for RETRY_VLM_*
  "user_summary_message": "Optional message for user on task end."
}}

If VLM fails to produce valid JSON repeatedly, guide it with simpler instructions or take blind action as a last resort.
Always aim for progress. Break down complex goals.
If VLM's 'global_thought' is missing many fields (filled with "N/A"), it might be struggling; simplify its instruction.
"""
    qwen_user_prompt_parts = []
    qwen_user_prompt_parts.append(f"Current Overall User Goal: {overall_user_goal}")
    
    # Rapport d'état du VLM
    vlm_report = current_vlm_status_report
    qwen_user_prompt_parts.append("\n--- VLM Frontend Status for its Last Instruction ---")
    if vlm_report.get("parsed_vlm_data"):
        qwen_user_prompt_parts.append(f"VLM Parsed Output:\n```json\n{json.dumps(vlm_report['parsed_vlm_data'], indent=2)}\n```")
    else:
        qwen_user_prompt_parts.append(f"VLM FAILED to provide usable/parsable JSON output for its last instruction.")
    if vlm_report.get("vlm_error_message"):
        qwen_user_prompt_parts.append(f"VLM Error/Warning Details: {vlm_report['vlm_error_message']}")
    if not vlm_report.get("parsed_vlm_data") and vlm_report.get("vlm_output_json_str"): # Montrer la sortie brute si le parsing a échoué
         qwen_user_prompt_parts.append(f"VLM Raw (unparsable/problematic) Output Snippet: ```\n{vlm_report['vlm_output_json_str'][:350]}...\n```")
    
    qwen_user_prompt_parts.append(f"Consecutive VLM Failures for Current VLM Instruction: {consecutive_vlm_failures_for_current_instruction_count}")

    # Résumé de l'historique des interactions pour Qwen
    history_summary_for_qwen_str = "\n\n--- Recent Interaction History (Your Past Qwen Decisions & VLM Outcomes, max 3 shown) ---"
    if full_interaction_history:
        for i, (enriched_hist_entry_json_str, _) in enumerate(reversed(full_interaction_history[-3:])): # Afficher les 3 dernières étapes enrichies
            try:
                entry_data = json.loads(enriched_hist_entry_json_str)
                qwen_prev_decision = entry_data.get("qwen_decision_obj", {})
                vlm_instr_given_to_vlm = entry_data.get("vlm_instruction_given", "N/A")
                
                summary_line = f"{i+1}. Your (Qwen) Prev Decision: '{qwen_prev_decision.get('decision_type','N/A')}' for VLM instruction: '{vlm_instr_given_to_vlm}'. "
                summary_line += f"Your Reason: '{qwen_prev_decision.get('reasoning','N/A')}'. "
                
                vlm_status_in_hist = entry_data.get("vlm_status_report", {})
                if vlm_status_in_hist.get("vlm_error_message"):
                    summary_line += f" VLM then had error: {vlm_status_in_hist['vlm_error_message']}. "
                elif entry_data.get("executed_vlm_action_sequence"):
                    summary_line += f" VLM actions then executed: {[a.get('action_type') for a in entry_data['executed_vlm_action_sequence']]}. "
                elif vlm_status_in_hist.get("parsed_vlm_data") and vlm_status_in_hist["parsed_vlm_data"].get("action_sequence"):
                     summary_line += f" VLM proposed actions: {[a.get('action_type') for a in vlm_status_in_hist['parsed_vlm_data']['action_sequence']]}. "
                history_summary_for_qwen_str += "\n" + summary_line
            except Exception as e_hist:
                history_summary_for_qwen_str += f"\n{i+1}. (Erreur de parsing de l'historique pour Qwen: {str(e_hist)})"
        qwen_user_prompt_parts.append(history_summary_for_qwen_str)
    else:
        qwen_user_prompt_parts.append("\n--- No interaction history yet for this overall task. ---")

    qwen_messages = [{"role": "system", "content": qwen_system_prompt}]
    
    # Préparer le contenu du message utilisateur (texte + image optionnelle pour Qwen-VL)
    current_user_message_content_list = []
    text_content_for_qwen = "\n".join(qwen_user_prompt_parts)
    current_user_message_content_list.append({"type": "text", "text": text_content_for_qwen})

    is_qwen_multimodal = "VL" in QWEN_MODEL_NAME_FOR_API.upper() # Heuristique simple
    if is_qwen_multimodal and image_base64_url:
        current_user_message_content_list.append({
            "type": "image_url",
            "image_url": {"url": image_base64_url, "detail": "high"}
        })
        print("[grey50]Image envoyée au Backend Qwen (car multimodal).[/grey50]")
    
    qwen_messages.append({"role": "user", "content": current_user_message_content_list})

    try:
        print(f"\nEnvoi de la requête au Backend Qwen (Modèle: {QWEN_MODEL_NAME_FOR_API})...")
        # Pour débogage intensif: print(f"Messages pour Qwen: {json.dumps(qwen_messages, indent=2)}")
        
        completion = api_client.chat.completions.create(
            model=QWEN_MODEL_NAME_FOR_API,
            messages=qwen_messages,
            max_tokens=1800, # Qwen pourrait avoir besoin de plus de tokens pour le JSON + raisonnement
            temperature=0.2,  # Température basse pour des décisions stratégiques plus fiables
            # response_format={"type": "json_object"} # À activer si votre endpoint Qwen le supporte bien et de manière fiable
        )
        qwen_response_str_raw = completion.choices[0].message.content
        print(f"[cyan]Réponse Brute du Backend Qwen:[/]\n{qwen_response_str_raw}")

        # Extraction robuste du JSON de la réponse de Qwen
        json_str_to_parse_qwen = None
        match_qwen = re.search(r"```json\s*(\{[\s\S]+?\})\s*```", qwen_response_str_raw, re.DOTALL)
        if match_qwen:
            json_str_to_parse_qwen = match_qwen.group(1)
            print("[grey50]JSON extrait du bloc Markdown de Qwen.[/grey50]")
        else:
            first_brace_qwen = qwen_response_str_raw.find('{')
            last_brace_qwen = qwen_response_str_raw.rfind('}')
            if first_brace_qwen != -1 and last_brace_qwen > first_brace_qwen:
                json_str_to_parse_qwen = qwen_response_str_raw[first_brace_qwen : last_brace_qwen + 1]
                print("[grey50]JSON extrait de Qwen par première/dernière accolade.[/grey50]")
            else:
                 print("[red]Erreur Qwen: Impossible d'extraire un objet JSON de la réponse de Qwen.[/red]")
                 raise ValueError("Aucune donnée JSON extractible de la réponse de Qwen.")

        qwen_decision = json.loads(json_str_to_parse_qwen)

        # Validation de base des clés obligatoires de la décision de Qwen
        required_qwen_keys = ["decision_type", "reasoning"]
        if not all(key in qwen_decision for key in required_qwen_keys):
            print(f"[red]Erreur Qwen: Réponse JSON de Qwen manque des clés obligatoires: {required_qwen_keys}. Réponse: {qwen_decision}[/red]")
            raise ValueError("Réponse JSON de Qwen manque des clés obligatoires.")
        # Validations supplémentaires basées sur decision_type
        dt = qwen_decision["decision_type"]
        if dt in ["EXECUTE_VLM_SEQUENCE", "EXECUTE_MODIFIED_SEQUENCE"] and not isinstance(qwen_decision.get("action_sequence_to_execute"), list) :
            print(f"[red]Erreur Qwen: Décision '{dt}' nécessite 'action_sequence_to_execute' (liste).[/red]")
            qwen_decision["action_sequence_to_execute"] = None # Neutraliser pour éviter erreur d'exécution
            # On pourrait changer decision_type en TASK_FAILED ici, mais laissons le flux principal gérer
        if dt == "RETRY_VLM_WITH_NEW_INSTRUCTION" and not qwen_decision.get("next_vlm_instruction"):
            print(f"[red]Erreur Qwen: Décision 'RETRY_VLM_WITH_NEW_INSTRUCTION' nécessite 'next_vlm_instruction'.[/red]")
            # Forcer un échec si l'instruction manque
            qwen_decision["decision_type"] = "TASK_FAILED"
            qwen_decision["reasoning"] += " (Erreur interne: next_vlm_instruction manquante pour RETRY)"
            
        return qwen_decision

    except Exception as e:
        print(f"[bold red]Erreur Critique durant l'appel au Backend Qwen ou le parsing de sa réponse: {e}[/bold red]")
        play_sound_feedback("error.wav")
        return { # Fournir un objet de décision d'échec par défaut
            "decision_type": "TASK_FAILED",
            "reasoning": f"Erreur d'interaction avec Qwen ou parsing de sa réponse: {str(e)}",
            "action_sequence_to_execute": None,
            "next_vlm_instruction": None,
            "user_summary_message": "Une erreur interne est survenue avec le module de raisonnement stratégique. Veuillez essayer une nouvelle tâche."
        }

# --- Boucle Principale de l'Agent ---
def main_agent_loop():
    print("[bold blue]Assistant de Navigation GUI (Architecture à Deux Niveaux)[/bold blue]")
    print(f"Utilisation VLM Frontend: API Base: {OPENAI_API_BASE_URL}, Modèle: {VLM_MODEL_NAME_FOR_API}")
    print(f"Utilisation LLM Backend (Qwen): API Base: {OPENAI_API_BASE_URL}, Modèle: {QWEN_MODEL_NAME_FOR_API}")
    print(f"Résolution d'écran: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    print("Tapez 'exit' ou 'quit' pour quitter.")

    overall_user_task = ""
    # interaction_history stocke des chaînes JSON de enriched_history_entry_data
    interaction_history = []
    current_task_step_count = 0 # Compteur pour la tâche utilisateur globale
    
    current_vlm_instruction = "" # Instruction spécifique pour le VLM
    consecutive_vlm_failures_for_current_instruction = 0 # Compte les échecs VLM pour l'instruction VLM actuelle

    while True:
        if not overall_user_task: # Si aucune tâche globale n'est en cours, en demander une nouvelle
            overall_user_task = Prompt.ask("\n[bold yellow]Quel est votre objectif global?[/bold yellow] (ou tapez 'exit'/'quit')")
            if overall_user_task.lower() in ["exit", "quit"]:
                print("Sortie de l'agent."); break
            if not overall_user_task: continue # Redemander si vide
            
            # Réinitialisation pour la nouvelle tâche globale
            interaction_history = []
            current_task_step_count = 0
            current_vlm_instruction = overall_user_task # Au début, l'instruction VLM est l'objectif utilisateur
            consecutive_vlm_failures_for_current_instruction = 0
            print(f"\n[bold magenta]Nouvel Objectif Global Utilisateur:[/] {overall_user_task}")
            play_sound_feedback("ask.wav")

        current_task_step_count += 1
        print(f"\n[bold_white on_blue]>>> Traitement Étape {current_task_step_count}/20 pour Objectif: '{overall_user_task}' <<[/bold_white on_blue]")
        print(f"Instruction VLM Actuelle: '{current_vlm_instruction}' (Échecs VLM sur cette instr.: {consecutive_vlm_failures_for_current_instruction})")

        if current_task_step_count > 20: # Limite de sécurité
            print(f"[bold red]Nombre maximum d'étapes ({20}) atteint pour l'objectif actuel. La tâche a échoué.[/bold red]")
            play_sound_feedback("error.wav")
            overall_user_task = ""; current_vlm_instruction = ""; interaction_history = []
            continue
        
        # --- Capture d'écran ---
        time.sleep(0.5) # Petite pause avant la capture pour laisser l'UI se stabiliser
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        screenshot_filename = f"step_{current_task_step_count}_{timestamp}.png"
        screenshot_path = os.path.join("agent_gui_screenshots_api", screenshot_filename)
        os.makedirs("agent_gui_screenshots_api", exist_ok=True)
        try:
            # S'assurer de capturer l'écran principal. Sur macOS, bbox=None par défaut pour l'écran principal.
            ImageGrab.grab(bbox=None, all_screens=False).save(screenshot_path)
            print(f"Capture d'écran sauvegardée: {screenshot_path}")
        except Exception as e:
            print(f"[red]Erreur lors de la capture d'écran: {e}[/red]"); play_sound_feedback("error.wav")
            time.sleep(2) # Attendre un peu avant de potentiellement réessayer
            # Pas d'image, Qwen sera informé via un rapport VLM vide/erroné
            # On pourrait forcer un appel à Qwen ici avec un statut d'erreur système, mais la boucle le fera naturellement.
            continue # Reboucler pour la prochaine tentative de step
        
        image_b64_url = image_to_base64_url(screenshot_path)
        if not image_b64_url:
            print("[red]Échec de l'encodage de la capture d'écran.[/red]"); play_sound_feedback("error.wav")
            time.sleep(2)
            continue

        # --- Appel au VLM Frontend ---
        # Construire l'historique résumé pour le prompt VLM (juste les actions VLM et les raisons de Qwen)
        vlm_history_summary_for_vlm_prompt = []
        for hist_entry_json_str, _ in interaction_history:
            try:
                entry = json.loads(hist_entry_json_str)
                vlm_instr = entry.get("vlm_instruction_given", "N/A")
                q_dec_reason = entry.get("qwen_decision_obj", {}).get("reasoning", "N/A")
                exec_actions = entry.get("executed_vlm_action_sequence")
                summary_line = f"Pour votre instruction '{vlm_instr}', Qwen a raisonné: '{q_dec_reason}'. "
                if exec_actions:
                    summary_line += f"Actions exécutées: {[a.get('action_type') for a in exec_actions]}."
                else:
                    summary_line += "Aucune action VLM n'a été exécutée ensuite (probablement une nouvelle instruction VLM)."
                vlm_history_summary_for_vlm_prompt.append(summary_line)
            except: pass # Ignorer les erreurs de parsing pour le résumé de l'historique VLM

        api_messages_for_vlm = build_messages_for_vlm_api(
            VLM_SYSTEM_PROMPT,
            current_vlm_instruction,
            image_b64_url,
            vlm_history_summary_for_vlm_prompt
        )
        
        vlm_raw_response_str = ""; parsed_vlm_data = None; vlm_api_or_parse_error_msg = None
        try:
            print(f"Envoi de la requête au VLM Frontend (Modèle: {VLM_MODEL_NAME_FOR_API})...")
            vlm_completion = client.chat.completions.create(
                model=VLM_MODEL_NAME_FOR_API, messages=api_messages_for_vlm,
                max_tokens=1200, temperature=0.01, # Température très basse pour le VLM
            )
            vlm_raw_response_str = vlm_completion.choices[0].message.content
            print(f"[grey50]Réponse Brute VLM: {vlm_raw_response_str[:300]}...[/grey50]") # Afficher un snippet
            parsed_vlm_data = parse_vlm_output_to_sequence(vlm_raw_response_str)
            if parsed_vlm_data is None: # Si parse_vlm_output_to_sequence retourne None
                vlm_api_or_parse_error_msg = "Le parsing du JSON VLM a échoué (voir logs VLM pour détails)."
        except Exception as e_vlm_api:
            print(f"[red]Erreur API VLM: {e_vlm_api}[/red]"); vlm_api_or_parse_error_msg = str(e_vlm_api); play_sound_feedback("error.wav")

        # Préparer le rapport d'état du VLM pour Qwen
        current_vlm_status_report_for_qwen = {
            "vlm_output_json_str": vlm_raw_response_str, # Toujours envoyer la sortie brute
            "parsed_vlm_data": parsed_vlm_data, # Peut être None
            "vlm_error_message": vlm_api_or_parse_error_msg # Message d'erreur si échec API/parsing
        }

        # Mettre à jour le compteur d'échecs VLM pour l'instruction actuelle
        if vlm_api_or_parse_error_msg or parsed_vlm_data is None:
            consecutive_vlm_failures_for_current_instruction += 1
            print(f"[orange_red1]Échec VLM pour l'instruction actuelle. Total échecs consécutifs: {consecutive_vlm_failures_for_current_instruction}[/orange_red1]")
        # Si le VLM a réussi à produire un JSON parsable, le compteur sera réinitialisé SEULEMENT si Qwen donne une NOUVELLE instruction VLM.
        # Si Qwen demande de réessayer la MÊME instruction VLM, le compteur n'est pas réinitialisé ici.

        # --- Appel au LLM Backend (Qwen) pour décision stratégique ---
        qwen_image_argument = image_b64_url if "VL" in QWEN_MODEL_NAME_FOR_API.upper() else None
        
        qwen_decision_obj = get_qwen_strategic_decision(
            client,
            overall_user_task,
            qwen_image_argument,
            current_vlm_status_report_for_qwen,
            interaction_history, # Historique complet des décisions/exécutions passées enrichies
            consecutive_vlm_failures_for_current_instruction # Passer le compteur d'échecs
        )

        # --- Traitement de la Décision de Qwen ---
        print(f"\n[bold_white on_purple]Décision du Backend Qwen ({qwen_decision_obj.get('decision_type')}):[/]")
        print(f"  [purple]Raisonnement de Qwen:[/purple] {qwen_decision_obj.get('reasoning')}")
        if qwen_decision_obj.get('next_vlm_instruction'):
            print(f"  [purple]Prochaine Instruction VLM (par Qwen):[/purple] {qwen_decision_obj.get('next_vlm_instruction')}")

        # Préparer l'entrée d'historique enrichie AVANT de modifier current_vlm_instruction
        enriched_history_entry_data = {
            "step_count": current_task_step_count,
            "overall_user_task": overall_user_task,
            "vlm_instruction_given_to_vlm": current_vlm_instruction, # L'instruction que le VLM vient de traiter
            "vlm_status_report": current_vlm_status_report_for_qwen, # Contient brut, parsé, erreur VLM
            "qwen_decision_obj": qwen_decision_obj,
            "executed_vlm_action_sequence": None # Sera rempli si une exécution a lieu
        }
        
        # Mise à jour de l'instruction VLM pour le prochain tour, et du compteur d'échecs
        next_vlm_instruction_from_qwen = qwen_decision_obj.get("next_vlm_instruction")
        qwen_decision_type = qwen_decision_obj.get("decision_type")

        if qwen_decision_type == "RETRY_VLM_WITH_NEW_INSTRUCTION":
            if next_vlm_instruction_from_qwen:
                if current_vlm_instruction == next_vlm_instruction_from_qwen:
                    # Qwen demande de réessayer la MÊME instruction VLM, le compteur d'échecs VLM n'est pas réinitialisé
                    print("[grey50]Qwen demande de réessayer la même instruction VLM. Le compteur d'échecs VLM persiste.[/grey50]")
                else:
                    # Qwen donne une NOUVELLE instruction VLM, réinitialiser le compteur d'échecs VLM
                    consecutive_vlm_failures_for_current_instruction = 0
                current_vlm_instruction = next_vlm_instruction_from_qwen
            else: # Erreur dans la logique de Qwen, il devrait fournir une instruction
                print("[red]Erreur: Qwen a demandé RETRY_VLM mais n'a pas fourni next_vlm_instruction. Forçage de l'échec de la tâche.[/red]")
                qwen_decision_type = "TASK_FAILED" # Forcer l'échec
                qwen_decision_obj["reasoning"] += " (Erreur interne Qwen: next_vlm_instruction manquante pour RETRY)"
        elif qwen_decision_type in ["EXECUTE_VLM_SEQUENCE", "EXECUTE_MODIFIED_SEQUENCE"]:
            # Après exécution, le VLM devra évaluer le nouvel état.
            # L'instruction pour le VLM sera de réévaluer l'état par rapport à l'objectif global.
            current_vlm_instruction = f"Actions (dirigées par Qwen) viennent d'être exécutées pour l'objectif '{overall_user_task}'. Évaluez le nouvel écran. Si l'objectif global n'est pas encore atteint, quel est le prochain sous-objectif VLM ? Si l'objectif global semble atteint, utilisez l'action FINISHED."
            consecutive_vlm_failures_for_current_instruction = 0 # Réinitialiser après une exécution réussie (ou tentative)
        elif qwen_decision_type in ["TASK_COMPLETED", "TASK_FAILED"]:
            pass # L'overall_user_task sera réinitialisé plus bas
        else: # Cas inconnu ou erreur de Qwen
            print(f"[red]Type de décision Qwen inconnu ou invalide: {qwen_decision_type}. Traitement comme échec de l'étape.[/red]")
            # Laisser l'instruction VLM telle quelle, Qwen devrait corriger au prochain tour.

        # Exécution des actions si Qwen l'a décidé
        actions_to_execute_from_qwen = qwen_decision_obj.get("action_sequence_to_execute")
        if qwen_decision_type in ["EXECUTE_VLM_SEQUENCE", "EXECUTE_MODIFIED_SEQUENCE"] and isinstance(actions_to_execute_from_qwen, list):
            print(f"[green]Exécution de {len(actions_to_execute_from_qwen)} micro-actions (dirigées par Qwen)...[/green]")
            enriched_history_entry_data["executed_vlm_action_sequence"] = actions_to_execute_from_qwen
            
            sequence_fully_executed_by_agent = True
            vlm_instruction_marked_finished_in_sequence = False # Si une action FINISHED du VLM est dans la séquence de Qwen
            for i, micro_action_from_qwen in enumerate(actions_to_execute_from_qwen):
                print(f"\n--- Exécution micro-action {i+1}/{len(actions_to_execute_from_qwen)} (Dirigée par Qwen) ---")
                m_action_type = micro_action_from_qwen.get("action_type")
                m_description = micro_action_from_qwen.get("description", "")
                action_step_executed_successfully = True

                if m_action_type == "CLICK": action_click(micro_action_from_qwen["position"], m_description)
                elif m_action_type == "DOUBLE_CLICK": action_double_click(micro_action_from_qwen["position"], m_description)
                elif m_action_type == "INPUT": action_input_text(micro_action_from_qwen["value"], micro_action_from_qwen.get("position"), m_description)
                elif m_action_type == "SCROLL": action_scroll(micro_action_from_qwen["direction"], m_description)
                elif m_action_type == "PRESS_ENTER": action_press_enter(m_description)
                elif m_action_type == "KEY_PRESS": action_step_executed_successfully = action_key_press(micro_action_from_qwen["keys"], m_description)
                elif m_action_type == "PAUSE": action_pause(micro_action_from_qwen["duration_seconds"], m_description)
                elif m_action_type == "FINISHED": # Si Qwen inclut une action FINISHED (venant du VLM ou générée par Qwen)
                    action_finished_vlm(micro_action_from_qwen.get("reason", "Instruction VLM/Qwen terminée."), m_description)
                    vlm_instruction_marked_finished_in_sequence = True
                else:
                    print(f"[red]Erreur: Type de micro_action inconnu dans la séquence de Qwen: {m_action_type}[/red]"); play_sound_feedback("error.wav")
                    sequence_fully_executed_by_agent = False; action_step_executed_successfully = False; break
                
                if action_step_executed_successfully is False: # Si une action comme KEY_PRESS retourne False
                    print(f"[red]La micro-action {m_action_type} a signalé un échec. Arrêt de la séquence dirigée par Qwen.[/red]"); play_sound_feedback("error.wav")
                    sequence_fully_executed_by_agent = False; break
                
                if m_action_type not in ["PAUSE", "FINISHED"]: # Petite pause après la plupart des actions GUI
                    time.sleep(0.7) # Ajustez cette valeur si nécessaire
            
            if not sequence_fully_executed_by_agent:
                print("[yellow]La séquence d'actions dirigée par Qwen n'a pas été entièrement exécutée à cause d'une erreur.[/yellow]")
            
            if vlm_instruction_marked_finished_in_sequence:
                 print("[grey50]Une action FINISHED était dans la séquence exécutée. Qwen évaluera la complétion globale de la tâche au prochain tour.[/grey50]")
                 # L'instruction VLM pour le prochain tour est déjà définie pour une réévaluation.
        
        elif qwen_decision_type in ["EXECUTE_VLM_SEQUENCE", "EXECUTE_MODIFIED_SEQUENCE"] and not isinstance(actions_to_execute_from_qwen, list) :
             print(f"[red]Erreur: Qwen a décidé '{qwen_decision_type}' mais 'action_sequence_to_execute' était manquante ou invalide. Pas d'exécution.[/red]")
             play_sound_feedback("error.wav")
             # L'historique contiendra cette décision problématique de Qwen.
        
        # Logique de fin de tâche basée sur la décision de Qwen
        if qwen_decision_type == "TASK_COMPLETED":
            summary_msg = qwen_decision_obj.get('user_summary_message', 'Tâche terminée avec succès.')
            print(f"[bold_green on_black]QWEN: OBJECTIF GLOBAL TERMINÉ! 🎉 Résumé: {summary_msg}[/bold_green on_black]")
            play_sound_feedback("task_completed.wav")
            overall_user_task = ""; current_vlm_instruction = ""; # Réinitialiser pour une nouvelle tâche
        
        elif qwen_decision_type == "TASK_FAILED":
            summary_msg = qwen_decision_obj.get('user_summary_message', "La tâche n'a pas pu être terminée.")
            print(f"[bold_red on_black]QWEN: OBJECTIF GLOBAL ÉCHOUÉ. 😥 Raison: {qwen_decision_obj.get('reasoning')}. Résumé: {summary_msg}[/bold_red on_black]")
            play_sound_feedback("error.wav")
            overall_user_task = ""; current_vlm_instruction = ""; # Réinitialiser
        
        elif qwen_decision_type == "RETRY_VLM_WITH_NEW_INSTRUCTION":
            play_sound_feedback("ask_2.wav") # Son pour indiquer que Qwen donne une nouvelle instruction au VLM
            # current_vlm_instruction est déjà mis à jour. La boucle continue.
        
        # Sauvegarder l'état enrichi de cette étape (sortie VLM + décision Qwen + actions exécutées)
        # La réponse brute de Qwen est stockée pour débogage si nécessaire.
        qwen_raw_response_for_history = qwen_decision_obj.get("raw_qwen_response_str_for_debug", json.dumps(qwen_decision_obj)) # Utiliser la réponse brute si disponible
        interaction_history.append((json.dumps(enriched_history_entry_data), vlm_raw_response_str + "\nQWEN_RAW_RESPONSE:\n" + qwen_raw_response_for_history ))

        if not overall_user_task: # Si la tâche a été marquée comme terminée ou échouée par Qwen
            print("--- Réinitialisation pour un nouvel objectif utilisateur global ---")
            interaction_history = [] # Effacer l'historique pour la nouvelle tâche utilisateur globale

if __name__ == "__main__":
    if AUDIO_ENABLED:
        audio_dir = "audio_feedback"
        os.makedirs(audio_dir, exist_ok=True)
        sample_rate = 16000 # Taux d'échantillonnage standard
        duration = 0.1 # Durée très courte pour les sons factices
        dummy_sound_data = [0.0] * int(duration * sample_rate) # Un silence court
        
        for f_name in ["ask.wav", "ok.wav", "ask_2.wav", "error.wav", "task_completed.wav"]:
            f_path = os.path.join(audio_dir, f_name)
            if not os.path.exists(f_path):
                try:
                    sf.write(f_path, dummy_sound_data, sample_rate, format='WAV', subtype='PCM_16')
                except Exception as e_audio:
                    print(f"[yellow]Impossible de créer le fichier audio factice {f_name}: {e_audio}[/yellow]")
    main_agent_loop()
