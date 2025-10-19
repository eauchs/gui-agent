

# Two-Layer GUI Navigation Agent

This project is a Graphical User Interface (GUI) automation agent for macOS, designed with a two-layer architecture to separate visual perception from strategic decision-making. It can understand high-level user objectives, analyze the screen, and execute sequences of actions (clicks, typing, keyboard shortcuts) to achieve those goals.

## 🤖 Architecture

The agent relies on two collaborating Language Models (LLMs):

1.  **The Frontend VLM (Perception):** A Vision Language Model that acts as the agent's "eyes." It receives a screenshot and a *specific* instruction from the supervisor. Its sole task is to analyze the image and propose a sequence of micro-actions (e.g., "click the button at position [x, y]," "type 'hello world'") in a strict JSON format.

      * **Model used (configurable):** `internvl3-8b-instruct`

2.  **The Backend LLM (Strategy):** A standard LLM that acts as the agent's "brain." It receives the user's *overall goal*, analyzes the VLM's output (or failure), evaluates if the plan is relevant, and makes the final decision to:

      * **Give a new instruction to the VLM** to refine the action.
      * **Approve the action sequence** proposed by the VLM for execution.
      * **Correct or propose its own action sequence** if the VLM is stuck or making repeated errors.
      * **Determine if the task is complete** or has failed.
      * **Model used (configurable):** `qwen/qwen3-8b`

This separation of concerns delegates the complex visual analysis task to a specialized model, while using a more "generalist" and strategic LLM for logic, error correction, and long-term planning.

## ✨ Features

  * **GUI Control:** Automates clicks, double-clicks, text input, scrolling, and keyboard shortcuts.
  * **Visual Feedback:** Displays overlays on-screen to indicate which action is currently being executed.
  * **Audio Feedback:** Plays sounds to notify of different stages (new task, success, error).
  * **Detailed Logging:** Saves screenshots, model decisions, and executed actions for each step, facilitating debugging.
  * **Flexible Configuration:** Models and API endpoints are configurable via environment variables.
  * **Robust Error Handling:** The supervisor (Qwen) can detect when the VLM fails and attempt to correct course or re-issue instructions.

## 🛠️ Installation

**Prerequisites**

  * Python 3.8+
  * A **local model server** compatible with the OpenAI API (e.g., **LM Studio**, **Ollama**). You will need to load the required VLM and LLM models onto it.
  * **macOS** (as `pyautogui` and `pynput` behaviors can vary by OS).

**Steps**

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/eauchs/gui-agent.git
    cd gui-agent
    ```

2.  **Create a virtual environment and activate it:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuration

The agent is configured using environment variables. You can set them in your terminal before running the script or use a `.env` file (with `pip install python-dotenv`).

1.  **API Endpoint:** Ensure your local server is running. The default URL is `http://localhost:1234/v1`.

    ```bash
    export OPENAI_API_BASE_URL="http://localhost:1234/v1"
    ```

2.  **Model Names:** These names must **exactly** match those loaded in your local server.

    ```bash
    # Model for visual analysis (VLM)
    export VLM_MODEL_NAME_FOR_API="internvl3-8b-instruct"

    # Model for strategy (LLM)
    export QWEN_MODEL_NAME_FOR_API="qwen/qwen3-8b"
    ```

## ▶️ Launch

Once dependencies are installed and environment variables are set, run the main script from your terminal:

```bash
python autonomous_gui_agent.py
```

The agent will prompt you to enter a global objective.

**Example Objectives:**

  * "Open Chrome, go to `https://www.google.com/search?q=google.com` and search for images of cute cats."
  * "Open the terminal, list the files in the current directory, then create a new folder called 'test\_agent'."
  * "Check if there are any system updates available in System Preferences."

To stop the agent, you can type `exit` or `quit` when prompted for an objective, or use `Ctrl+C` in the terminal.

## 📝 Generated Files & Folders

During execution, the agent automatically creates:

  * `agent_gui_screenshots_api/`: A folder containing a screenshot for each step of the task.
  * `agent_gui_screenshots_api/detailed_interaction_log.txt`: A highly detailed log file, recording prompts, raw model responses, and executed actions. Useful for debugging.
  * `audio_feedback/`: Contains the sound files generated for audio feedback.
