import torch
import numpy as np
import matplotlib.pyplot as plt
from config import device

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