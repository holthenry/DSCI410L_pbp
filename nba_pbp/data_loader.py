import pandas as pd
import os

def load_tracking():
    path = os.path.join(os.path.dirname(__file__), "..", "temp_game", "0021500492.json")
    return pd.read_json(path)

def load_pbp():
    path = os.path.join(os.path.dirname(__file__), "..", "temp_game", "pbp.csv")
    return pd.read_csv(path)