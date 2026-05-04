import torch
import numpy as np
import matplotlib.pyplot as plt
from reconstruction_par_IA_GAN.seq2seq_python_project.config import device

def plot_results(model, test_x, test_y, num_plots=4):
    model.eval()
    with torch.no_grad():
        test_x_tensor = torch.FloatTensor(test_x).to(device)
        predictions = model(test_x_tensor).cpu().numpy()
        
    seq_past = test_x.shape[1]
    seq_future = test_y.shape[1]
    input_dim = test_x.shape[2]
    
    x_past = np.arange(seq_past)
    x_future = np.arange(seq_past, seq_past + seq_future)
    
    for i in range(min(num_plots, test_x.shape[0])):
        plt.figure(figsize=(10, 4))
        for dim in range(input_dim):
            plt.plot(x_past, test_x[i, :, dim], color='blue', label='Entrée (Passé)' if dim==0 else "")
            plt.plot(x_future, test_y[i, :, dim], color='green', label='Attendu (Futur)' if dim==0 else "")
            plt.plot(x_future, predictions[i, :, dim], color='orange', linestyle='--', label='Prédiction' if dim==0 else "")
        plt.title(f"Exemple de prédiction #{i+1}")
        plt.legend()
        plt.show()

def plot_results_imputation(model, test_x, test_y, hole_length, hole_start, num_plots=4):
    model.eval()
    with torch.no_grad():
        test_x_tensor = torch.FloatTensor(test_x).to(device)
        predictions = model(test_x_tensor).cpu().numpy()
        
    input_len = test_x.shape[1]
    
    part1_x = np.arange(hole_start)
    part2_x = np.arange(hole_start, hole_start + hole_length)
    part3_x = np.arange(hole_start + hole_length, input_len + hole_length)
    
    for i in range(min(num_plots, test_x.shape[0])):
        plt.figure(figsize=(10, 4))
        plt.plot(part1_x, test_x[i, :hole_start, 0], color='blue', label='Entrée (Avant)')
        plt.plot(part3_x, test_x[i, hole_start:, 0], color='cyan', label='Entrée (Après)')
        plt.plot(part2_x, test_y[i, :, 0], color='green', label='Attendu (Trou)')
        plt.plot(part2_x, predictions[i, :, 0], color='orange', linestyle='--', label='Prédiction')
        plt.title(f"Exemple d'imputation #{i+1}")
        plt.legend(loc='lower left')
        plt.show()

def plot_results_v5(model, test_x, test_y, test_mask, num_plots=4):
    model.eval()
    num_samples = min(num_plots, test_x.shape[0])
    with torch.no_grad():
        test_x_tensor = torch.FloatTensor(test_x[:num_samples]).to(device)
        predictions = model(test_x_tensor).cpu().numpy()
        
    max_ctx = 300
    for i in range(num_samples):
        plt.figure(figsize=(10, 4))
        hole_len = int(np.sum(test_mask[i, :, 0]))
        
        plt.plot(np.arange(0, max_ctx), test_x[i, :max_ctx, 0], color='blue', alpha=0.5, label='Entrée (Gauche + Pad)')
        
        x_right = np.arange(max_ctx + hole_len, max_ctx + hole_len + max_ctx)
        plt.plot(x_right, test_x[i, max_ctx:, 0], color='cyan', alpha=0.5, label='Entrée (Droite + Pad)')
        
        x_hole = np.arange(max_ctx, max_ctx + hole_len)
        plt.plot(x_hole, test_y[i, :hole_len, 0], color='green', label=f'Attendu (len={hole_len})')
        plt.plot(x_hole, predictions[i, :hole_len, 0], color='orange', linestyle='--', label='Prédiction')
        
        plt.title(f"Imputation Dynamique (Trou={hole_len}) | Zoom ±50")
        plt.legend(loc='lower left')
        plt.xlim(max_ctx - 50, max_ctx + hole_len + 50)
        plt.show()

def plot_results_v7(model, test_x, test_y, test_mask, max_ctx=250, num_plots=4):
    model.eval()
    num_samples = min(num_plots, test_x.shape[0])
    with torch.no_grad():
        test_x_tensor = torch.FloatTensor(test_x[:num_samples]).to(device)
        predictions = model(test_x_tensor).cpu().numpy()
        
    for i in range(num_samples):
        plt.figure(figsize=(10, 4))
        
        hole_len = int(np.sum(test_mask[i, :, 0]))
        
        plt.plot(np.arange(0, max_ctx), test_x[i, :max_ctx, 0], color='blue', alpha=0.5, label='Entrée (Gauche + Pad)')
        
        x_right = np.arange(max_ctx + hole_len, max_ctx + hole_len + max_ctx)
        plt.plot(x_right, test_x[i, max_ctx:, 0], color='cyan', alpha=0.5, label='Entrée (Droite + Pad)')
        
        x_hole = np.arange(max_ctx, max_ctx + hole_len)
        plt.plot(x_hole, test_y[i, :hole_len, 0], color='green', label=f'Attendu (len={hole_len})')
        plt.plot(x_hole, predictions[i, :hole_len, 0], color='orange', linestyle='--', label='Prédiction')
        
        plt.title(f"Imputation Auto-supervisée (Trou={hole_len}) | Zoom ±50")
        plt.legend(loc='lower left')
        plt.xlim(max_ctx - 50, max_ctx + hole_len + 50)
        plt.show()

def reconstruct_and_plot_real_signal(model, time_arr, sig_nan, mean, std, max_ctx=250, max_hole=70):
    model.eval()
    sig_reconstructed = sig_nan.copy()
    
    is_nan = np.isnan(sig_nan)
    padded = np.concatenate(([0], is_nan.view(np.int8), [0]))
    diffs = np.diff(padded)
    nan_starts = np.where(diffs == 1)[0]
    nan_ends = np.where(diffs == -1)[0]
    
    for st, en in zip(nan_starts, nan_ends):
        hole_len = en - st
        if hole_len > max_hole:
            continue
            
        ctx_l_st = max(0, st - max_ctx)
        ctx_r_en = min(len(sig_nan), en + max_ctx)
        
        part1 = sig_reconstructed[ctx_l_st : st]
        part3 = sig_nan[en : ctx_r_en]
        
        part1 = np.nan_to_num(part1, nan=np.nanmean(sig_nan)) 
        part3 = np.nan_to_num(part3, nan=np.nanmean(sig_nan))
        
        ctx_l_len = len(part1)
        ctx_r_len = len(part3)
        
        x_ = np.zeros((1, max_ctx * 2, 1))
        x_[0, max_ctx - ctx_l_len : max_ctx, 0] = (part1 - mean) / std
        x_[0, max_ctx : max_ctx + ctx_r_len, 0] = (part3 - mean) / std
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_).to(device)
            pred = model(x_tensor).cpu().numpy()[0, :, 0]
            
        pred_denorm = (pred[:hole_len] * std) + mean
        sig_reconstructed[st:en] = pred_denorm

    plt.figure(figsize=(14, 5))
    zoom = 1000
    
    plt.plot(time_arr[:zoom], sig_nan[:zoom], color='blue', label='Signal Capteur Brut (Trous visibles)', linewidth=1.5)
    
    reconstructed_only = sig_reconstructed.copy()
    reconstructed_only[~np.isnan(sig_nan)] = np.nan
    
    plt.plot(time_arr[:zoom], reconstructed_only[:zoom], color='orange', label='Reconstruction IA', linewidth=2.0)
    
    plt.title("Reconstruction en chaine d'un signal industriel (Zoom sur 1000 pts)")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()