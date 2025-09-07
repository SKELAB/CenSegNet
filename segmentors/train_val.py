import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_map50(preds, targets, threshold=0.5):
    """
    mAP50 (mean Average Precision with IoU threshold of 0.5)
    
    :param preds: (B, 1, H, W)
    :param targets: (B, 1, H, W)
    :param threshold: IoU 0.5
    :return: mAP50
    """
    preds = (preds > threshold).cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()

    # AP
    ap = average_precision_score(targets, preds)
    
    return ap


def calculate_map50_95(preds, targets, thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    mAP50-95 (mean Average Precision from threshold IoU=0.5 to IoU=0.95)
    """
    all_map_scores = []
    for threshold in thresholds:
        pred = (preds > threshold).cpu().numpy()
        target = targets.cpu().numpy()
        map_score = average_precision_score(target.flatten(), pred.flatten())
        all_map_scores.append(map_score)

    mAP50_95 = np.mean(all_map_scores)
    return mAP50_95

def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs,save_folder, save_path):
    best_loss = 1e+6
    best_model_weights = None

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        model.train()
        total_loss = 0

        # train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        # for i, (imgs, masks) in train_loader_tqdm:
        for imgs, masks in tqdm(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)  # (B, 1, H, W)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        total_precision = 0
        total_recall = 0
        total_mAP50 = 0
        total_mAP50_95 = 0  
        total_samples = len(val_loader)

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)  # (B, 1, H, W)

                loss = criterion(outputs, masks)
                total_val_loss += loss.item()


        avg_val_loss = total_val_loss / total_samples

        if best_loss >= avg_val_loss:
            best_loss = avg_val_loss
            best_model_weights = model.state_dict()
            
            save_path_full = os.path.join(save_folder, f'best_{epoch}' + save_path)
            torch.save(best_model_weights, save_path_full)
            print(f"Best model updated with loss: {best_loss:.4f}")
    
    final_model_path = os.path.join(save_folder, f'final_' + save_path)
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Best loss: {best_loss:.4f}")
    model.load_state_dict(best_model_weights)
    return model
