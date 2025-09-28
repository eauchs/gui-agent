
# Agent de Navigation GUI à Deux Niveaux

Ce projet est un agent d'automatisation d'interface graphique (GUI) pour macOS, conçu avec une architecture à deux niveaux pour séparer la perception visuelle de la prise de décision stratégique. Il est capable de comprendre des objectifs utilisateur de haut niveau, d'analyser l'écran et d'exécuter des séquences d'actions (clics, frappe de texte, raccourcis clavier) pour atteindre ces objectifs.

## 🤖 Architecture

L'agent repose sur deux modèles de langage (LLM) qui collaborent :

1.  **Le VLM Frontend (Perception)** : Un modèle de langage multimodal (Vision Language Model) qui agit comme les "yeux" de l'agent. Il reçoit une capture d'écran et une instruction *spécifique* du superviseur. Sa seule tâche est d'analyser l'image et de proposer une séquence de micro-actions (par exemple, "cliquer sur le bouton à la position [x, y]", "taper 'hello world'") dans un format JSON strict.

      * **Modèle utilisé (configurable)** : `internvl3-8b-instruct`

2.  **Le LLM Backend (Stratégie)** : Un modèle de langage standard qui agit comme le "cerveau" de l'agent. Il reçoit l'objectif global de l'utilisateur, analyse la sortie (ou l'échec) du VLM, évalue si le plan est pertinent et prend la décision finale.

      * **Donner une nouvelle instruction au VLM** pour affiner l'action.
      * **Approuver la séquence d'actions** proposée par le VLM pour exécution.
      * **Corriger ou proposer sa propre séquence d'actions** si le VLM est bloqué ou fait des erreurs répétées.
      * **Déterminer si la tâche est terminée** ou a échoué.
      * **Modèle utilisé (configurable)** : `qwen/qwen3-8b`

Cette séparation permet de confier la tâche complexe d'analyse visuelle à un modèle spécialisé, tout en utilisant un LLM plus "généraliste" et stratégique pour la logique, la correction d'erreurs et la planification à long terme.

## ✨ Fonctionnalités

  * **Contrôle de l'interface graphique** : Automatise les clics, doubles-clics, la saisie de texte, le défilement et les raccourcis clavier.
  * **Feedback visuel** : Affiche des superpositions (overlays) à l'écran pour indiquer quelle action est en cours d'exécution.
  * **Feedback audio** : Joue des sons pour notifier les différentes étapes (nouvelle tâche, succès, erreur).
  * **Logging détaillé** : Enregistre les captures d'écran, les décisions des modèles et les actions exécutées pour chaque étape, facilitant le débogage.
  * **Configuration flexible** : Les modèles et les points d'accès API sont configurables via des variables d'environnement.
  * **Gestion des erreurs robuste** : Le superviseur (Qwen) peut détecter lorsque le VLM échoue et tenter de corriger le tir ou de reformuler les instructions.

## 🛠️ Installation

### Prérequis

  * **Python 3.8+**
  * Un **serveur de modèles local** compatible avec l'API OpenAI (par exemple, [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/)). Vous devrez y charger les modèles VLM et LLM requis.
  * **macOS** (car `pyautogui` et `pynput` ont des comportements qui peuvent varier selon l'OS).

### Étapes

1.  **Clonez le dépôt :**

    ```bash
    git clone https://votre-url-de-depot.git
    cd nom-du-repertoire
    ```

2.  **Créez un environnement virtuel et activez-le :**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Installez les dépendances Python :**

    ```bash
    pip install -r requirements.txt
    ```

    *Note : Si vous n'avez pas de fichier `requirements.txt`, installez les paquets manuellement :*

    ```bash
    pip install openai pyautogui Pillow pynput rich sounddevice soundfile numpy
    ```

## ⚙️ Configuration

L'agent est configuré à l'aide de variables d'environnement. Vous pouvez les définir dans votre terminal avant de lancer le script ou utiliser un fichier `.env`.

1.  **Point d'accès API** : Assurez-vous que votre serveur local est en cours d'exécution. L'URL par défaut est `http://localhost:1234/v1`.

    ```bash
    export OPENAI_API_BASE_URL="http://localhost:1234/v1"
    ```

2.  **Nom des modèles** : Les noms doivent correspondre **exactement** à ceux chargés dans votre serveur local.

    ```bash
    # Modèle pour l'analyse visuelle (VLM)
    export VLM_MODEL_NAME_FOR_API="internvl3-8b-instruct"

    # Modèle pour la stratégie (LLM)
    export QWEN_MODEL_NAME_FOR_API="qwen/qwen3-8b"
    ```

## ▶️ Lancement

Une fois les dépendances installées et les variables d'environnement configurées, lancez le script principal depuis votre terminal :

```bash
python nom_du_script.py
```

L'agent vous demandera de saisir un objectif global.

### Exemples d'objectifs

  * "Ouvre Chrome, va sur https://www.google.com/search?q=google.com et cherche des images de chats mignons."
  * "Ouvre le terminal, liste les fichiers dans le répertoire actuel, puis crée un nouveau dossier appelé 'test\_agent'."
  * "Vérifie s'il y a des mises à jour système disponibles dans les Préférences Système."

Pour arrêter l'agent, vous pouvez taper `exit` ou `quit` lorsque vous êtes invité à saisir un objectif, ou utiliser `Ctrl+C` dans le terminal.

## 📝 Fichiers et Dossiers générés

Pendant son exécution, l'agent crée automatiquement :

  * `agent_gui_screenshots_api/` : Un dossier contenant une capture d'écran pour chaque étape de la tâche.
  * `agent_gui_screenshots_api/detailed_interaction_log.txt` : Un fichier journal très détaillé, enregistrant les prompts, les réponses brutes des modèles et les actions exécutées. Utile pour le débogage.
  * `audio_feedback/` : Contient les fichiers sonores générés pour le feedback audio.
