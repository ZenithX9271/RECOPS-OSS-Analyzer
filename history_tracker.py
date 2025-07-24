import os

# Use a subdirectory for history files to avoid permission issues
HISTORY_DIR = "data"
HISTORY_FILE = os.path.join(HISTORY_DIR, "analysis_history.txt")

def ensure_history_dir():
    """Ensure the history directory exists (create if needed)."""
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)

def save_history_entry(filename):
    ensure_history_dir()
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(filename + "\n")
    except Exception as e:
        print(f"[History Error] Could not write entry: {e}")

def get_past_runs(max_entries=10):
    ensure_history_dir()
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            entries = [line.strip() for line in f.readlines()][-max_entries:]
        return entries[::-1]  # Show latest first
    except Exception as e:
        print(f"[History Error] Could not read history: {e}")
        return []
