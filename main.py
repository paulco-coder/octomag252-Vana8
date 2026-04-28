import torch

# Importation des fonctions extraites depuis le script intermédiaire
from fonctions_notebook import load_data, train_model, evaluate_model

def main():
    """
    Fonction principale orchestrant tout le pipeline de main.ipynb en version pure Python (.py).
    """
    print("=== DÉMARRAGE DU PIPELINE GAN 1D (BATCH MODE) ===")
    
    # 1. Paramètres pour entraînement complet (identiques au dernier main.ipynb)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil de calcul alloué : {DEVICE}")

    WINDOW_SIZE = 512       # Fenêtre optimisée pour 10-15 cycles
    EPOCHS = 100            # Temps de convergence du WGAN-GP
    BATCH_SIZE = 64         # Stabilisation des patchs sur GPU
    NUM_SAMPLES = 5000      # Échantillonnage étendu
    
    fichier_entree = 'signaux_entree_dataset.pt'
    
    # 2. Exécution du chargement (Cellule Data)
    try:
        dataset_tensor = load_data(fichier_entree)
    except AssertionError as msg:
        print(msg)
        return
        
    # 3. Exécution de l'entraînement (Cellule Train)
    trained_generator = train_model(
        dataset_tensor=dataset_tensor,
        window_size=WINDOW_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES,
        device=DEVICE
    )
    
    # 4. Exécution de l'évaluation (Cellule Inférence & Plot)
    evaluate_model(
        trained_generator=trained_generator,
        dataset_tensor=dataset_tensor,
        window_size=WINDOW_SIZE,
        device=DEVICE
    )
    
    print("=== PIPELINE TERMINÉ AVEC SUCCÈS ===")

if __name__ == "__main__":
    main()