import numpy as np
from reconstruction_par_IA_GAN.seq2seq_python_project.config import device
from reconstruction_par_IA_GAN.seq2seq_python_project.models import Seq2Seq, BiImputationModel
from reconstruction_par_IA_GAN.seq2seq_python_project.data_utils import generate_data_v1, generate_data_v2, generate_data_v3, generate_data_v4, generate_data_v5, generate_real_sensor_signals, build_self_supervised_dataset
from reconstruction_par_IA_GAN.seq2seq_python_project.train_utils import train_model, train_model_v5, train_model_v6_fft
from reconstruction_par_IA_GAN.seq2seq_python_project.plot_utils import plot_results, plot_results_imputation, plot_results_v5, plot_results_v7, reconstruct_and_plot_real_signal

def main():
    print(f"Démarrage de la simulation sur: {device}")
    
    # ------------------ Exercice 1 ------------------
    print("\n--- Exercice 1 : Prédiction déterministe ---")
    X_1, Y_1 = generate_data_v1(1000, 10)
    X_1 = (X_1 - np.mean(X_1)) / (np.std(X_1) + 1e-8)
    Y_1 = (Y_1 - np.mean(X_1)) / (np.std(X_1) + 1e-8)
    split_idx = int(0.85 * len(X_1))
    train_x_1, val_x_1 = X_1[:split_idx], X_1[split_idx:]
    train_y_1, val_y_1 = Y_1[:split_idx], Y_1[split_idx:]

    model_1 = Seq2Seq(input_dim=2, hidden_dim=12, output_dim=2, future_seq_len=10, num_layers=2).to(device)
    train_model(model_1, train_x_1, train_y_1, val_x_1, val_y_1, epochs=15, batch_size=100, lr=0.005)
    plot_results(model_1, val_x_1, val_y_1, num_plots=1)

    # ------------------ Exercice 2 ------------------
    print("\n--- Exercice 2 : Superposition de fréquences ---")
    X_2, Y_2 = generate_data_v2(5000, 15)  # Echantillon un peu réduit
    mean_2, std_2 = np.mean(X_2), np.std(X_2) + 1e-8
    X_2, Y_2 = (X_2 - mean_2) / std_2, (Y_2 - mean_2) / std_2
    split_idx_2 = int(0.85 * len(X_2))
    train_x_2, val_x_2 = X_2[:split_idx_2], X_2[split_idx_2:]
    train_y_2, val_y_2 = Y_2[:split_idx_2], Y_2[split_idx_2:]

    model_2 = Seq2Seq(input_dim=1, hidden_dim=35, output_dim=1, future_seq_len=15, num_layers=2).to(device)
    train_model(model_2, train_x_2, train_y_2, val_x_2, val_y_2, epochs=5, batch_size=50, lr=0.005)
    plot_results(model_2, val_x_2, val_y_2, num_plots=1)

    # ------------------ Exercice 4 ------------------
    print("\n--- Exercice 4 : Imputation ---")
    hole_length_4, hole_start_4 = 30, 30
    X_4, Y_4 = generate_data_v4(10000, 90, hole_start_4, hole_length_4)
    mean_4, std_4 = np.mean(X_4), np.std(X_4) + 1e-8
    X_4, Y_4 = (X_4 - mean_4) / std_4, (Y_4 - mean_4) / std_4
    split_idx_4 = int(0.85 * len(X_4))
    train_x_4, val_x_4 = X_4[:split_idx_4], X_4[split_idx_4:]
    train_y_4, val_y_4 = Y_4[:split_idx_4], Y_4[split_idx_4:]

    model_4 = BiImputationModel(input_dim=1, hidden_dim=64, output_dim=1, hole_len=hole_length_4, num_layers=2).to(device)
    train_model(model_4, train_x_4, train_y_4, val_x_4, val_y_4, epochs=10, batch_size=100, lr=0.001, wd=0.0)
    plot_results_imputation(model_4, val_x_4, val_y_4, hole_length_4, hole_start_4, num_plots=1)

    # ------------------ Exercice 6 ------------------
    print("\n--- Exercice 6 : FFT Loss sur signaux complexes variables ---")
    X_5, Y_5, Mask_5 = generate_data_v5(10000)
    mean_5, std_5 = np.mean(X_5), np.std(X_5) + 1e-8
    X_5, Y_5 = (X_5 - mean_5) / std_5, (Y_5 - mean_5) / std_5
    split_idx_5 = int(0.85 * len(X_5))
    
    train_x_5, val_x_5 = X_5[:split_idx_5], X_5[split_idx_5:]
    train_y_5, val_y_5 = Y_5[:split_idx_5], Y_5[split_idx_5:]
    train_m_5, val_m_5 = Mask_5[:split_idx_5], Mask_5[split_idx_5:]

    model_6 = BiImputationModel(input_dim=1, hidden_dim=64, output_dim=1, hole_len=50, num_layers=2).to(device)
    train_model_v6_fft(model_6, train_x_5, train_y_5, train_m_5, val_x_5, val_y_5, val_m_5, epochs=10, batch_size=128, lr=0.001, alpha=0.5)
    plot_results_v5(model_6, val_x_5, val_y_5, val_m_5, num_plots=1)

    # ------------------ Exercice 7 ------------------
    print("\n--- Exercice 7 : Cas Pratique Hautes Fréquences avec Trous Périodiques ---")
    sensor_signals = generate_real_sensor_signals(n_signals=15, base_duration_sec=1.0)
    MAX_CTX_7 = 250
    MAX_HOLE_7 = 70
    X_7, Y_7, Mask_7 = build_self_supervised_dataset(sensor_signals, max_ctx=MAX_CTX_7, max_hole=MAX_HOLE_7)
    
    mean_7 = np.mean(X_7)
    std_7 = np.std(X_7) + 1e-8
    X_7 = (X_7 - mean_7) / std_7
    Y_7 = (Y_7 - mean_7) / std_7

    split_idx_7 = int(0.85 * len(X_7))
    train_x_7, val_x_7 = X_7[:split_idx_7], X_7[split_idx_7:]
    train_y_7, val_y_7 = Y_7[:split_idx_7], Y_7[split_idx_7:]
    train_m_7, val_m_7 = Mask_7[:split_idx_7], Mask_7[split_idx_7:]

    model_7 = BiImputationModel(input_dim=1, hidden_dim=96, output_dim=1, hole_len=MAX_HOLE_7, num_layers=3).to(device)
    train_model_v6_fft(model_7, train_x_7, train_y_7, train_m_7, val_x_7, val_y_7, val_m_7, epochs=30, batch_size=128, lr=0.001, alpha=0.5)
    
    plot_results_v7(model_7, val_x_7, val_y_7, val_m_7, max_ctx=MAX_CTX_7, num_plots=3)
    
    t0, sig0_nan, sig0_vrai = sensor_signals[0]
    reconstruct_and_plot_real_signal(model_7, t0, sig0_nan, mean=mean_7, std=std_7, max_ctx=MAX_CTX_7, max_hole=MAX_HOLE_7)

if __name__ == "__main__":
    main()