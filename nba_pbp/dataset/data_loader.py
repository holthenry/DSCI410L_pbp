import io
import os
import json
import random
import tempfile
import requests
import py7zr
import pandas as pd
import numpy as np

def define_possessions(pbp):
    pbp = pbp.sort_values(['GAME_ID', "EVENTNUM"]).reset_index(drop=True)
    team_changed = pbp['PLAYER1_TEAM_ID'] != pbp['PLAYER1_TEAM_ID'].shift(1)
    game_changed = pbp['GAME_ID'] != pbp['GAME_ID'].shift(1)
    new_pos_flag = team_changed | game_changed
    pbp["possession_id"] = new_pos_flag.cumsum()
    return pbp

def parse_score(score_str):
    if pd.isna(score_str):
        return None
    away, home = score_str.split(" - ")
    return int(away), int(home)

def pts_scored(pbp):
    pbp = define_possessions(pbp)
    scores = pbp["SCORE"].apply(parse_score)
    pbp["away_score"] = scores.apply(lambda x: x[0] if x is not None else None)
    pbp["home_score"] = scores.apply(lambda x: x[1] if x is not None else None)
    
    pbp["away_score"] = pbp.groupby("GAME_ID")["away_score"].ffill().fillna(0).astype(int)
    pbp["home_score"] = pbp.groupby("GAME_ID")["home_score"].ffill().fillna(0).astype(int)

    pos_ends = pbp.groupby(["GAME_ID", "possession_id"])[["away_score", "home_score"]].last().reset_index()
    pos_ends["total_score"] = pos_ends["away_score"] + pos_ends["home_score"]
    
    prev_total_score = pos_ends.groupby("GAME_ID")["total_score"].shift(1).fillna(0)
    pos_ends["calculated_points"] = (pos_ends["total_score"] - prev_total_score).astype(int)
    
    pos_to_pts = dict(zip(pos_ends["possession_id"], pos_ends["calculated_points"]))
    pbp["points"] = pbp["possession_id"].map(pos_to_pts)
    return pbp

def extract_nba_sequence(moments_list, home_id, away_id):
    sequence = []
    for moment in moments_list:
        if not moment or len(moment) < 6 or not isinstance(moment[5], list):
            continue
            
        game_clock = moment[2]
        shot_clock = moment[3] if moment[3] is not None else 24.0
        entities = moment[5]
        
        ball = [e for e in entities if isinstance(e, list) and len(e) >= 5 and e[0] == -1]
        if not ball:
            continue  
        ball_x, ball_y, ball_z = ball[0][2], ball[0][3], ball[0][4]
        
        home_players = [p for p in entities if isinstance(p, list) and len(p) >= 5 and p[0] == home_id]
        away_players = [p for p in entities if isinstance(p, list) and len(p) >= 5 and p[0] == away_id]
        
        home_players = sorted(home_players, key=lambda x: x[1])[:5]
        away_players = sorted(away_players, key=lambda x: x[1])[:5]
        
        if len(home_players) < 5 or len(away_players) < 5:
            continue  
            
        frame_features = [game_clock, shot_clock, ball_x, ball_y, ball_z]
        for hp in home_players:
            frame_features.extend([hp[2], hp[3]])  
        for ap in away_players:
            frame_features.extend([ap[2], ap[3]])  
            
        sequence.append(frame_features)
    return np.array(sequence)

def pad_tracking_sequences(sequences, maxlen=400, num_features=25):
    padded_out = np.zeros((len(sequences), maxlen, num_features), dtype=np.float32)
    for idx, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        length = min(len(seq), maxlen)
        padded_out[idx, :length, :] = seq[:length, :]
    return padded_out

def run_github_download_pipeline(user="linouk23", repo="NBA-Player-Movements", folder_path="data/2016.NBA.Raw.SportVU.Game.Logs", 
                                 output_dir="./games", files=None, shuffle=False):

    pbp_df = load_pbp()
    os.makedirs(output_dir, exist_ok=True)

    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder_path}"
    api_response = requests.get(api_url)

    repo_contents = api_response.json()
    target_files = [f for f in repo_contents if f['name'].endswith('.7z')]
    
    if shuffle:
        random.shuffle(target_files)
        
    processed_count = 0
    
    for file_info in target_files:
        if files is not None and processed_count >= files:
            break
            
        filename = file_info['name']
        game_file_id = filename.replace('.7z', '')
        
        if os.path.exists(f"{output_dir}/{game_file_id}_X.npy"):
            processed_count += 1
            continue
            
        raw_download_url = file_info['download_url']
        response = requests.get(raw_download_url, timeout=45)
        file_stream = io.BytesIO(response.content)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with py7zr.SevenZipFile(file_stream, mode='r') as archive:
                archive.extractall(path=tmpdir)
            
            extracted_files = os.listdir(tmpdir)
            json_filenames = [f for f in extracted_files if f.endswith('.json')]
            if not json_filenames:
                continue
                
            target_json_path = os.path.join(tmpdir, json_filenames[0])
            with open(target_json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            raw_game_id = json_data.get('gameid') or json_data.get('gameId')
            if not raw_game_id and 'events' in json_data and len(json_data['events']) > 0:
                raw_game_id = json_data['events'][0].get('gameid') or json_data['events'][0].get('gameId')
                
            if not raw_game_id:
                continue
                
            clean_game_id_int = int(str(raw_game_id).strip().lstrip('0'))
            
            if 'events' not in json_data or len(json_data['events']) == 0:
                continue
                
            json_events_df = pd.DataFrame(json_data['events'])
            id_col = 'eventId' if 'eventId' in json_events_df.columns else 'eventid'
            json_events_df['EVENTNUM_INT'] = json_events_df[id_col].astype(int)
            
            pbp_df['GAME_ID_INT'] = pbp_df['GAME_ID'].astype(str).str.strip().str.lstrip('0').astype(int)
            pbp_df['EVENTNUM_INT'] = pbp_df['EVENTNUM'].astype(int)
            
            game_pbp_lookup = pbp_df[pbp_df['GAME_ID_INT'] == clean_game_id_int][['EVENTNUM_INT', 'points']].copy()
            if game_pbp_lookup.empty:
                continue
                
            json_events_df = json_events_df.merge(game_pbp_lookup, on='EVENTNUM_INT', how='left')
            json_events_df['points'] = json_events_df['points'].fillna(0.0)
            
            game_sequences = []
            final_points = []
            
            first_event = json_data['events'][0]
            home_id = first_event['home']['teamid']
            away_id = first_event['visitor']['teamid']
            
            for _, row in json_events_df.iterrows():
                moments_list = row['moments']
                if moments_list is None or not isinstance(moments_list, list) or len(moments_list) == 0:
                    continue
                    
                event_matrix = extract_nba_sequence(moments_list, home_id, away_id) 
                if event_matrix.shape[0] > 0:
                    game_sequences.append(event_matrix)
                    final_points.append(row['points'])
                    
            if len(game_sequences) > 0:
                y_pts_array = np.array(final_points, dtype=np.float32)
                X_padded = pad_tracking_sequences(game_sequences, maxlen=400, num_features=25)
                
                np.save(f"{output_dir}/{game_file_id}_X.npy", X_padded)
                np.save(f"{output_dir}/{game_file_id}_y_pts.npy", y_pts_array)
                processed_count += 1

    print(f"saved {processed_count} games")

def load_tracking_demo():
    path = os.path.join(os.path.dirname(__file__), "..", "temp_game", "0021500492.json")
    return pd.read_json(path)

def load_pbp_demo():
    path = os.path.join(os.path.dirname(__file__), "..", "temp_game", "pbp.csv")
    return pd.read_csv(path)

def load_pbp():
    pbp_url = "https://github.com/sumitrodatta/nba-alt-awards/raw/main/Historical/PBP%20Data/2015-16_pbp.csv"
    pbp_df = pd.read_csv(pbp_url, dtype={'GAME_ID': str})
    return pts_scored(pbp_df)