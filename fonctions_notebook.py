import os
import torch
import matplotlib.pyplot as plt
import sys

# Ajout du dossier courant au path pour importer le module src correctement
sys.path.append('.')

from src.dataset import SignalDataset
from src.networks import Generator, Discriminator
from src.train import train_gan
from src.inference import reconstruct_signal

def load_data(fichier_entree='signaux_entree_dataset.pt'):
    """
    Équivalent de la cellule de préparation et chargement des données.
    Charge le tenseur PyTorch d'entraînement.
    """
    if os.path.exists(fichier_entree):
        dataset_tensor = torch.load(fichier_entree)
        print(f"Chargement réussi. Shape: {dataset_tensor.shape}")
        return dataset_tensor
    else:
        raise FileNotFoundError(f"Erreur : Le fichier '{fichier_entree}' n'existe pas. Veuillez exécuter 'generation_signaux.ipynb' (ou son équivalent) auparavant.")

def train_model(dataset_tensor, window_size, epochs, batch_size, num_samples, device):
    """
    Équivalent de la cellule d'entraînement.
    Prépare le Dataset, instancie les réseaux locaux et lance la boucle de WGAN.
    """
    print("Préparation du dataset (extraction d'échantillons valides et auto-supervision)...")
    dataset = SignalDataset(dataset_tensor, window_size=window_size, num_samples=num_samples)

    # Note: On garde les features ajustés pour un entraînement local ou GPU
    gen = Generator(in_channels=2, out_channels=1, features=8)  
    disc = Discriminator(in_channels=1, features=4)

    print("Démarrage de l'entraînement auto-supervisé...")
    trained_generator = train_gan(
        dataset=dataset, 
        generator=gen, 
        discriminator=disc, 
        epochs=epochs, 
        batch_size=batch_size, 
        d_updates=5, 
        g_updates=1,
        device=device
    )
    return trained_generator

def evaluate_model(trained_generator, dataset_tensor, window_size, device):
    """
    Équivalent de la dernière cellule d'inférence.
    Reconstruit un signal amputé complet et affiche le comparatif original vs généré.
    """
    # Test sur le SIGNAL n°1 (index 0) du tenseur d'origine (qui contient les `NaN`)
    original_signal_with_nans = dataset_tensor[0].clone()

    print("Démarrage de l'inférence pour restaurer le signal de test...")
    reconstructed = reconstruct_signal(
        generator=trained_generator, 
        single_signal=original_signal_with_nans, 
        window_size=window_size, 
        device=device
    )

    print("Génération du graphique comparatif...")
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    # Plotting sans interpoler les NaN -> des "trous" physiques apparaissent
    plt.plot(original_signal_with_nans.numpy()[:1000], label='Original avec Pertes (NaN)', color='red')
    plt.title("Signal Origine capteur (0-1000 premiers points)")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    # Signal réparé par le GAN (les NaN sont tous remplis)
    plt.plot(reconstructed.numpy()[:1000], label='Signal Reconstruit (GAN)', color='green')
    plt.title("Restauration par le Modèle U-Net GAN")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()