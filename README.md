# Greenova: Energy-Adaptive AI for Onboard Satellite Inference

**IASTAM 6.0 Technical Challenge — Track 1, Problem 2 (TUNSA Collaboration)**

## Contexte

Les satellites n'ont pas d'alimentation électrique stable : chaque orbite alterne entre une phase de lumière solaire (~60 min, énergie haute) et une phase d'éclipse (~30 min, énergie en baisse). Une IA embarquée doit continuer à fonctionner dans les deux cas, sans provoquer de défaillance de la mission.

## Le mécanisme Greenova

Greenova traite l'énergie non pas comme une contrainte fixe, mais comme une variable dynamique qui pilote en temps réel le comportement du modèle d'IA — le "Energy-Inference Pendulum" :

| État d'énergie | Comportement du modèle |
|---|---|
| Haute (State 1) | Réseau complet déployé — précision maximale |
| Nominale (State 2) | Couches intermédiaires court-circuitées — compromis précision/vitesse |
| Critique (State 3) | Modèle le plus superficiel, fréquence d'inférence réduite — survie du cœur système |

Implémenté avec un MobileNetV2 fine-tuné sur EuroSAT (classification de paysages satellite, 10 classes), doté de 3 sorties (early exits) correspondant aux 3 états, et d'un simulateur d'orbite qui bascule automatiquement entre elles selon le niveau d'énergie simulé.

## Résultats clés — Compression du modèle

Deux approches de quantification ont été testées pour réduire la taille de chaque état :

| Méthode | Réduction de taille | Perte de précision |
|---|---|---|
| Quantification statique naïve (PTQ) | jusqu'à 71% | jusqu'à -20 points (inacceptable) |
| **Quantization-Aware Training (QAT)** | **54% à 71%** | **-1.6 à -3.2 points seulement** |

La PTQ naïve dégrade fortement la précision sur MobileNetV2, un phénomène documenté sur les architectures depthwise-separable. Le QAT permet d'obtenir la même compression tout en préservant une précision quasi intacte — rendant la compression viable pour un déploiement embarqué réel.

## Structure du dépôt

- `greenova_notebook.ipynb` — notebook complet : chargement des données EuroSAT, fine-tuning MobileNetV2, mécanisme multi-sorties Greenova, simulateur d'énergie orbitale, quantification PTQ vs QAT

