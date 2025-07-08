# === history_tracker.py ===
import os

HISTORY_FILE = "analysis_history.txt"


def save_history_entry(filename):
    with open(HISTORY_FILE, "a") as f:
        f.write(filename + "\n")


def get_past_runs(max_entries=10):
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        entries = [line.strip() for line in f.readlines()][-max_entries:]
    return entries[::-1]  # show latest first
