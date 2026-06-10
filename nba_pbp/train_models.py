import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn

from model.model import ExpectedPointsLSTM
from dataset.data_loader import run_github_download_pipeline

class NBATrackingDataset(Dataset):
    def __init__(self, processed_dir="./games"):
        self.processed_dir = processed_dir
        search_path = os.path.join(processed_dir, "*_X.npy")
        x_files = glob.glob(search_path)   
        self.game_ids = [os.path.basename(f).replace("_X.npy", "") for f in x_files]

    def __len__(self):
        return len(self.game_ids)

    def __getitem__(self, idx):
        game_id = self.game_ids[idx]
        
        X = np.load(os.path.join(self.processed_dir, f"{game_id}_X.npy"))
        y = np.load(os.path.join(self.processed_dir, f"{game_id}_y_pts.npy"))
        y = y.astype(np.float32)
        y_normalized = y / 3.0
        
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=-1)

        X_norm = np.copy(X)
        X_norm[:, :, 0] /= 720.0  # game clock
        X_norm[:, :, 1] /= 24.0   # shot clock
        X_norm[:, :, 2] /= 94.0   # ball X
        X_norm[:, :, 3] /= 50.0   # ball Y
        X_norm[:, :, 4] /= 15.0   # ball Z
        
        for i in range(5, 25, 2):
            X_norm[:, :, i]   /= 94.0
            X_norm[:, :, i+1] /= 50.0
            
        X_final = np.clip(X_norm, 0.0, 1.0)

        true_lengths = np.sum(np.any(X_final != 0.0, axis=2), axis=1)

        X_cropped = X_final[:, :, 2:]

        X_tensor = torch.from_numpy(X_cropped).float()
        y_tensor = torch.from_numpy(y).float()
        lengths_tensor = torch.from_numpy(true_lengths).long()
        
        X_tensor = torch.nan_to_num(X_tensor, nan=0.0)
        y_tensor = torch.nan_to_num(y_tensor, nan=0.0)

        return X_tensor, y_tensor, lengths_tensor

def combine_game_batches(batch):

    X_list = [item[0] for item in batch]
    y_list = [item[1] for item in batch]
    len_list = [item[2] for item in batch]
    
    # Concatenate along axis 0 to create continuous play vectors
    X_out = torch.cat(X_list, dim=0)
    y_out = torch.cat(y_list, dim=0)
    lengths_out = torch.cat(len_list, dim=0)
    
    return X_out, y_out, lengths_out
    
def get_data_loaders(dir='./games', batch_size=2, output_dir='./games', files=None, shuffle=False):

    run_github_download_pipeline(output_dir=output_dir, files=files, shuffle=shuffle)
    master_dataset = NBATrackingDataset(processed_dir=dir)
    
    total_games = len(master_dataset)
    train_size = int(0.80 * total_games)
    val_size = total_games - train_size

    generator = torch.Generator().manual_seed(42)

    train_subset, val_subset = random_split(
        master_dataset, 
        [train_size, val_size], 
        generator=generator
    )
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=combine_game_batches
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=combine_game_batches
    )
    
    return train_loader, val_loader


def training_loop(train_loader, val_loader, epochs=20, lr=1e-4):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    
    # Instantiate the model architecture
    model = ExpectedPointsLSTM(input_dim=23, hidden_dim=128, output_dim=1).to(device)
    criterion = nn.SmoothL1Loss(reduction='none') 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

    losses = {
        "train_loss": [],
        "val_loss": []
    }
    
    for epoch in range(epochs):
        
        model.train()
        train_running_loss = 0.0
        train_batches = 0
        
        for batch_X, batch_y, batch_lengths in train_loader:
            if batch_X.size(0) == 0: 
                continue
                
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            outputs = model(batch_X, batch_lengths)
            
            max_len = batch_X.size(1)
            mask = torch.arange(max_len, device=device)[None, :] < batch_lengths.to(device)[:, None]
            mask = mask.unsqueeze(-1)
            
            current_batch_size = outputs.shape[0]
            batch_y_dynamic = torch.zeros_like(outputs) 
            
            for i in range(current_batch_size):
                true_length = batch_lengths[i].item()
                final_score = batch_y[i].item() 
                
                if final_score > 0 and true_length > 0:
                    # Linearly ramp up to 2.0 or 3.0 at the final frame
                    ramp = torch.linspace(0.0, final_score, steps=true_length, device=device)
                    batch_y_dynamic[i, :true_length, 0] = ramp
                else:
                    batch_y_dynamic[i, :true_length, 0] = 0.0
            
            raw_loss = criterion(outputs, batch_y_dynamic)    
            loss_weights = torch.where(batch_y_dynamic > 0, 30.0, 1.0)
            masked_loss = raw_loss * mask.float() * loss_weights
            loss = masked_loss.sum() / mask.sum()
    
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item()
            train_batches += 1
            
        model.eval()
        val_running_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y, batch_lengths in val_loader:
                if batch_X.size(0) == 0: 
                    continue
                    
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X, batch_lengths)
                
                max_len = batch_X.size(1)
                mask = torch.arange(max_len, device=device)[None, :] < batch_lengths.to(device)[:, None]
                mask = mask.unsqueeze(-1)
                
                v_batch_size = outputs.shape[0]
                val_y_dynamic = torch.zeros_like(outputs)
                
                for i in range(v_batch_size):
                    true_length = batch_lengths[i].item()
                    final_score = batch_y[i].item()
                    
                    if final_score > 0 and true_length > 0:
                        ramp = torch.linspace(0.0, final_score, steps=true_length, device=device)
                        val_y_dynamic[i, :true_length, 0] = ramp
                    else:
                        val_y_dynamic[i, :true_length, 0] = 0.0
                
                raw_loss = criterion(outputs, val_y_dynamic)
                loss_weights = torch.where(val_y_dynamic > 0, 30.0, 1.0)
                masked_loss = raw_loss * mask.float() * loss_weights
                
                val_loss = masked_loss.sum() / mask.sum()
                val_running_loss += val_loss.item()
                val_batches += 1
                
        epoch_train_loss = train_running_loss / train_batches if train_batches > 0 else 0.0
        epoch_val_loss = val_running_loss / val_batches if val_batches > 0 else 0.0

        losses["train_loss"].append(epoch_train_loss)
        losses["val_loss"].append(epoch_val_loss)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
        
    print("done")
    
    return model, losses

if __name__ == "__main__":
    run_github_download_pipeline(files=5, shuffle=True)
    run_training_loop(processed_dir="./games", epochs=20, batch_size=2, lr=1e-4)