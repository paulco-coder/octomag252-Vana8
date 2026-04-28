# Guide de Tuning et Paramétrage du GAN 1D

Ce document centralise les différents hyperparamètres du projet et explique comment les ajuster pour améliorer la qualité de la reconstruction, accélérer l'entraînement ou adapter le modèle à de nouvelles données.

---

## 1. Paramètres d'Entraînement Globaux (`main.ipynb`)

Ces paramètres contrôlent la boucle d'apprentissage principale et l'utilisation des ressources (GPU/CPU).

*   **`WINDOW_SIZE` (ex: 512)** : La taille de la fenêtre temporelle analysée par le réseau.
    *   *Symptôme si trop petit* : Le modèle ne voit pas assez de cycles pour comprendre la composante sinusoïdale de base.
    *   *Symptôme si trop grand* : Consomme beaucoup de VRAM sur le GPU, et le `PatchGAN` a du mal à se concentrer sur les détails locaux.
*   **`BATCH_SIZE` (ex: 64)** : Le nombre d'échantillons traités en parallèle.
    *   *Recommandation* : Un batch élevé aide énormément à stabiliser l'apprentissage d'un WGAN-GP. Montez-le aussi haut que la VRAM de votre GPU le permet (128, 256...).
*   **`EPOCHS` (ex: 100)** : Le nombre de passages complets sur les données. Les GAN sont souvent lents à converger. Observez la courbe de la fonction de perte locale (L1) et spectrale ; si elles descendent encore, il faut continuer.
*   **`NUM_SAMPLES` (ex: 5000)** : Nombre de sous-segments générés artificiellement pour constituer "une epoch". Plus ce nombre est grand, plus le modèle voit de variations de trous.

---

## 2. Capacité de l'Architecture (`src/networks.py` via `main.ipynb`)

Dans `main.ipynb`, lors de l'instanciation des réseaux, le paramètre `features` définit la complexité (nombre de filtres) de l'architecture.

*   **Générateur (U-Net)** : `features=8` (ou 16, 32, 64)
    *   *À augmenter si* : Le modèle produit un signal trop "plat" ou lisse et échoue à reproduire le bruit haute fréquence complexe.
    *   *À diminuer si* : Le modèle mémorise le bruit par cœur (surapprentissage) ou manque de mémoire GPU.
*   **Discriminateur (PatchGAN)** : `features=4` (ou 8, 16)
    *   *Attention* : Dans un WGAN, le discriminateur (le *critique*) doit être assez puissant pour bien évaluer le générateur. Un rapport de puissance équilibré (ex: Gen 32 / Disc 16) est conseillé.

---

## 3. Dynamique WGAN et Optimiseurs (`src/train.py`)

Les GAN sont instables par nature. Le WGAN-GP règle ce problème mathématiquement, mais dépend de ses propres réglages :

*   **Ratio d'update (`d_updates=5`, `g_updates=1`)** :
    *   Pour le Wasserstein GAN, le Critique (Discriminateur) doit TOUJOURS être plus "intelligent" que le Générateur à un instant *t*. Un ratio de 5 contre 1 est le standard théorique validé.
*   **Pondération des fonctions de perte (Lambdas)** :
    *   `lambda_gp = 10` : Poids de la pénalité de gradient (Loi de Lipschitz). **À ne pas modifier globalement**, c'est une constante mathématique très robuste.
    *   `lambda_l1 = 100` : Poids de la "ressemblance" visuelle du trou comblé par rapport au signal réel. Si le modèle génère des ondes avec la bonne fréquence mais un volume (amplitude) faux, augmentez cette valeur.
    *   `lambda_spectral = 1.0` : Poids de préservation de la FFT (Transformée de Fourier). Si la sinusoïde reconstruite est déphasée ou a une mauvaise période, repoussez ce lambda (ex: 10.0).
*   **Learning Rate (Optimiseur Adam)** :
    *   Par défaut : `lr=0.0002` avec momentum `betas=(0.5, 0.9)`. C'est le standard pour les GAN. Si le modèle diverge brusquement (loss qui tend vers l'infini), abaissez le lr (ex: `0.0001` ou `0.00005`).

---

## 4. Phase d'Inférence (`src/inference.py`)

La reconstruction en production se fait par fenêtre glissante :

*   **Astuce de recouvrement (`step = window_size // 4`)** : L'algorithme se décale d'un quart de fenêtre à chaque itération. Cela permet d'avoir toujours beaucoup de contexte avant ET après la zone de `NaN` à réparer (puisqu'on veut éviter de combler un trou situé sur les bords de la fenêtre). Si vous avez des problèmes ou des "clics" entre les raccords de réparation, diminuez ce pas (ex: `window_size // 8`).
