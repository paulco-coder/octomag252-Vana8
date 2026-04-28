# Etude
***

# Rapport d'Analyse : Reconstruction de Signaux de Capteurs Haute Fréquence par Architecture GAN

## 1. Contexte et Synthèse du Projet

Le projet vise à imputer (reconstruire) des segments manquants dans des séries temporelles issues de capteurs à haute fréquence. L'objectif n'est pas de retrouver la valeur exacte d'origine, mais de générer un signal statistiquement et visuellement indiscernable du signal réel, en respectant la continuité temporelle et fréquentielle.

L'approche envisagée repose sur un Réseau Antagoniste Génératif (GAN), composé d'un générateur chargé d'imputer les données manquantes à partir du contexte adjacent, et d'un discriminateur devant distinguer les zones réelles des zones reconstruites.

**Paramètres Physiques et Numériques :**

| Paramètre | Valeur / Plage | Implication Technique |
| :--- | :--- | :--- |
| **Fréquence d'échantillonnage ($f_s$)** | $100 000 \text{ Hz}$ | Résolution temporelle très fine. |
| **Vitesse de rotation ($f_{rot}$)** | $1000 \text{ à } 3000 \text{ Hz}$ | Dynamique extrêmement rapide, cycles courts. |
| **Points par cycle ($N = \frac{f_s}{f_{rot}}$)** | $\approx 33 \text{ à } 100 \text{ pts/tour}$ | Les fenêtres de contexte sont extrêmement réduites. |
| **Taux de perte (par cycle)** | $5\% \text{ à } 25\%$ | Les "trous" représentent entre $1.5$ et $25$ points par rotation. |
| **Nature du signal** | De quasi-sinusoïdal à bruit blanc | L'algorithme doit généraliser sur la phase (sinus) et sur les statistiques (bruit). |

---

## 2. Analyse Critique des Contraintes et de la Proposition

L'idée d'utiliser une architecture GAN est tout à fait pertinente pour le critère d'**indiscernabilité**. Les modèles génératifs excellent pour créer des textures ou des bruits cohérents, là où des méthodes classiques (comme l'interpolation spline ou le filtrage de Kalman) échoueraient à reproduire un aspect "bruit blanc" naturel et produiraient un lissage artificiel facilement détectable.

Cependant, la proposition soulève plusieurs défis majeurs qui nécessitent une adaptation de l'approche standard :

### 2.1. L'absence de segments "purs" de grande taille
La contrainte la plus forte du projet est l'absence de signaux continus inaltérés (les trous étant présents sur toute la longueur). Un GAN classique a besoin d'échantillons réels pour apprendre au discriminateur ce qu'est la "réalité". 
* **Critique :** Si le discriminateur ne voit que des signaux troués ou reconstruits, il ne saura jamais à quoi ressemble un signal parfait.
* **Solution :** Il est impératif d'utiliser une approche d'**apprentissage auto-supervisé par masquage artificiel**. Puisque les trous représentent au maximum 25% du signal, les 75% restants entre deux trous consécutifs sont intacts. L'algorithme doit extraire ces sous-segments valides (par exemple, des fenêtres de 20 à 70 points consécutifs), y créer de *nouveaux* trous artificiels de taille similaire, et s'entraîner à les reconstruire. Le discriminateur comparera alors le sous-segment avec le trou artificiel reconstruit au sous-segment valide d'origine.

### 2.2. Le biais de cyclicité du discriminateur
L'inquiétude concernant le fait que le discriminateur repère les zones reconstruites simplement par leur position périodique est très fondée. Les réseaux de neurones sont des "paresseux intelligents" : s'ils peuvent minimiser leur erreur en comptant simplement le temps entre chaque trou plutôt qu'en analysant la texture du signal, ils le feront.
* **Solution :** Il faut appliquer un *random cropping* (découpage aléatoire) asynchrone. Lors de l'entraînement, les fenêtres fournies au discriminateur ne doivent jamais commencer à la même phase de rotation. De plus, les trous artificiels créés pour l'entraînement doivent être insérés à des positions aléatoires dans la fenêtre d'observation, décorrélant ainsi la position du trou de la vitesse de rotation de la machine.

### 2.3. L'instabilité de l'entraînement et l'équilibrage des réseaux
L'ajustement dynamique de la vitesse d'apprentissage (Learning Rate) pour éviter qu'un réseau ne prenne le dessus sur l'autre est une méthode empirique souvent instable et difficile à régler sur de nouveaux signaux.
* **Solution :** Il est préférable de se tourner vers une fonction de perte plus stable mathématiquement, telle que celle du **Wasserstein GAN avec pénalité de gradient (WGAN-GP)**. Le WGAN-GP empêche naturellement le discriminateur de devenir "trop parfait" en forçant ses gradients à respecter une contrainte Lipschitz, offrant ainsi un retour d'information continu et utile au générateur sans nécessiter de micro-gestion des taux d'apprentissage.

---

## 3. Architecture Recommandée

Pour répondre aux exigences de continuité temporelle et fréquentielle, l'architecture suivante est préconisée.



### 3.1. Le Générateur : U-Net 1D ou WaveNet
Le générateur doit comprendre le contexte global tout en générant des détails locaux haute fréquence. 
* Un **U-Net 1D** (réseau convolutif avec des connexions résiduelles entre les couches de compression et de décompression) est idéal. Il prend en entrée le signal tronqué (les NaN étant remplacés par des zéros) et un masque binaire indiquant la position des trous.
* Les convolutions doivent être **causales et dilatées** (style WaveNet) si le contexte temporel est large, mais compte tenu du faible nombre de points par cycle ($N < 100$), des convolutions 1D standards avec des noyaux de taille moyenne (e.g., 5 ou 7) suffiront pour capturer un cycle entier.

### 3.2. Le Discriminateur : PatchGAN 1D
Au lieu d'un discriminateur qui émet un seul jugement (Vrai/Faux) pour tout le signal, un **PatchGAN 1D** évalue l'authenticité de *chaque sous-section* du signal indépendamment. 
* Cela force le réseau à se concentrer sur la continuité locale et la texture fréquentielle plutôt que sur la structure globale ou la périodicité des erreurs.



### 3.3. Conception de la Fonction de Perte (Loss)
Pour s'assurer que l'algorithme respecte la composante cyclique (quasi-sinusoïdale) autant que la composante stochastique (bruit), la fonction de perte du générateur doit être composite :
$$\mathcal{L}_{Total} = \lambda_{1}\mathcal{L}_{Adversariale} + \lambda_{2}\mathcal{L}_{Spectrale}$$
* **$\mathcal{L}_{Adversariale}$** : Assure l'indiscernabilité (le bruit blanc aura la bonne texture).
* **$\mathcal{L}_{Spectrale}$** : Une perte calculée sur la Transformée de Fourier Rapide (FFT) des zones générées par rapport au contexte. Cela garantit que les fréquences dominantes (liées à la rotation $f_{rot}$ de 1000 à 3000 Hz) et leurs harmoniques ne subissent pas de rupture de phase lors de la reconstruction.

---

## 4. Bilan et Viabilité

La reconstruction de ce signal est un défi de micro-imputation temporel. La force de l'approche proposée réside dans la brièveté des zones à reconstruire (souvent moins de 10 points consécutifs). 

En basculant sur un paradigme d'entraînement **auto-supervisé** (sur les fragments sains existants) combiné à un **WGAN-GP 1D** évaluant des patchs aléatoires, le risque de surapprentissage sur la périodicité de la machine est neutralisé. Le système sera capable de généraliser à des signaux inconnus, car il apprendra la physique de la continuité d'une onde et la statistique du bruit local, plutôt que la forme globale d'un essai spécifique.

***

# Plan d'action

Le rapport suivant détaille le plan d'action séquentiel pour implémenter cette solution auto-supervisée en utilisant le framework PyTorch.

***

# Plan d'Action : Implémentation du Modèle GAN 1D sous PyTorch

L'implémentation de ce système nécessite une structuration rigoureuse, divisée en cinq phases distinctes, allant de la préparation des tenseurs jusqu'à la phase d'inférence.

## Phase 1 : Pipeline de Données (Data Engineering)

La première étape consiste à créer un flux d'alimentation robuste capable de générer les données d'entraînement à la volée. Il convient de développer une classe personnalisée héritant de `torch.utils.data.Dataset`.

* **Extraction dynamique (Random Crop) :** Lors de l'appel de la méthode `__getitem__`, le code doit sélectionner aléatoirement une fenêtre de taille fixe (par exemple, 512 ou 1024 points) à l'intérieur d'un fragment sain identifié par le script préalable.
* **Génération de masques artificiels :** Sur cette fenêtre saine, le pipeline doit générer un tenseur binaire (le masque) simulant une perte de 5 à 25% du signal. Les trous doivent être placés de manière aléatoire pour éviter tout biais périodique.
* **Préparation des Tenseurs :** Le dataset doit retourner trois éléments encapsulés via un `DataLoader` :
    1.  Le signal d'origine complet (la cible ou "Ground Truth").
    2.  Le masque binaire (1 pour les données valides, 0 pour les trous).
    3.  Le signal masqué (où les zones de trous sont forcées à 0).

## Phase 2 : Construction des Architectures (Modélisation)

Il est nécessaire d'implémenter les deux réseaux de neurones en créant des classes héritant de `torch.nn.Module`.

* **Le Générateur (U-Net 1D) :** * Utiliser des couches `nn.Conv1d` pour l'encodeur (compression) et `nn.ConvTranspose1d` (ou une interpolation suivie d'une convolution) pour le décodeur.
    * Le réseau doit accepter 2 canaux en entrée (le signal masqué + le masque binaire) et sortir 1 canal (le signal reconstruit).
    * Intégrer des *Skip Connections* (concaténation temporelle) entre les couches de même résolution de l'encodeur et du décodeur pour préserver les hautes fréquences.
* **Le Discriminateur (PatchGAN 1D) :**
    * Empiler des couches `nn.Conv1d` avec un pas (stride) supérieur à 1 pour réduire progressivement la dimension temporelle.
    * Contrairement à un classifieur classique, la dernière couche ne doit pas être un scalaire unique, mais un vecteur (carte de caractéristiques 1D) où chaque valeur juge l'authenticité d'un sous-segment (patch) du signal.
    * Utiliser `nn.LeakyReLU` pour l'activation afin de prévenir la mort des gradients.

## Phase 3 : Définition des Fonctions de Perte (Loss Functions)

Cette phase est critique pour stabiliser l'entraînement et forcer le respect des fréquences de rotation de la machine.

* **Perte Adversariale (WGAN-GP) :** * Implémenter la fonction de perte de Wasserstein (la moyenne des sorties du discriminateur).
    * Coder la "Gradient Penalty" : il faut créer des échantillons interpolés entre les signaux réels et générés, les passer dans le discriminateur, et utiliser `torch.autograd.grad` pour s'assurer que la norme du gradient reste proche de 1.
* **Perte de Continuité (L1) :** Utiliser `nn.L1Loss` calculée *uniquement* sur les zones masquées (en multipliant la sortie par l'inverse du masque) pour forcer le générateur à se rapprocher de la forme d'onde locale.
* **Perte Spectrale :** Utiliser `torch.fft.rfft` pour passer les signaux (réels et générés) dans le domaine fréquentiel, puis calculer l'erreur absolue entre les spectres d'amplitude. Cela garantira la préservation des harmoniques liées à la rotation de la machine.

## Phase 4 : Boucle d'Apprentissage (Training Loop)

La dynamique d'entraînement d'un WGAN exige une orchestration spécifique entre les deux réseaux.

* **Optimiseurs :** Instancier deux optimiseurs distincts (ex. `torch.optim.Adam`), avec des paramètres spécifiques recommandés pour les WGAN (souvent un *learning rate* faible et des betas ajustés comme `betas=(0.0, 0.9)`).
* **Ratio d'entraînement :** Le discriminateur (le "critique" dans un WGAN) doit être entraîné plus fréquemment que le générateur. L'architecture classique prévoit 5 mises à jour du discriminateur pour 1 mise à jour du générateur.
* **Monitoring :** Intégrer un outil de suivi (comme TensorBoard via `torch.utils.tensorboard` ou Weights & Biases) pour tracer l'évolution des pertes (Adversariale, L1, Spectrale) et visualiser les formes d'ondes reconstruites à intervalles réguliers.

## Phase 5 : Phase d'Inférence (Déploiement)

Une fois le modèle entraîné, il faut développer la logique pour traiter les signaux réels tronqués issus des capteurs.

* **Fenêtrage glissant :** Créer une fonction qui parcourt le signal réel à l'aide d'une fenêtre de la même taille que celle utilisée à l'entraînement.
* **Reconstruction ciblée :** Mettre le modèle en mode évaluation (`model.eval()`) et désactiver le calcul des gradients (`with torch.no_grad():`). Pour chaque fenêtre contenant des valeurs `NaN` (préalablement converties en 0 avec le masque correspondant), utiliser le générateur pour prédire le signal.
* **Fusion spatiale :** Remplacer les valeurs `NaN` du signal d'origine par les valeurs correspondantes prédites par le générateur, en appliquant éventuellement un léger lissage (cross-fade) aux frontières de la reconstruction pour éviter les clics audibles/fréquentiels.