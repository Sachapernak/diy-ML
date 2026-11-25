# Résultats du 24/11/2026

### Script utilisé :

- mlp_mnist.py

### Durée d'exécution : 
- 70 minutes environ

## Comparaison des modèles et des optimiseurs

Le script permet de tester des combinaisons de MLP et d'optimiseurs sur un meme nombre d'epoch et sur les mêmes seeds, avec les mêmes données à chaque fois.

On retient ensuite les meilleures epochs pour chaque seed, et leurs métriques (accuracy et loss de validation)

### Interprétations des résultats

Pour ce problème, l'architecture compte plus que l'optimiseur. On gagne 5 points de pourcentage en passant d'une 
regression logistique à un MLP (92% -> 97,5%) tandis que les différent optimiseur ne font gagner au mieux pas plus quelques 
dixièmes de point, en général. Cependant, aucun n'arrive a passer la barre des 98%. Le plafond vient probablement de 
l'architecture, et en passant à un CNN, je devrais pouvoir facilement dépasser ces résultats.

Les différents optimiseurs se valent a peut près avec mes hyperparameters, avec AdEMAMix très légèrement devant les 
autres (jusqu'a environ +0.2-0.3 points). Cependant, sur les MLP AdamW et AdEMAMix convergent nettement plus rapidement 
en terme d'epochs, que SGD et Momentum et peuvent finir par faire du sur-apprentissage alors que SGD et Momentum 
n'ont pas encore convergé. 

Le MLP "classique" avec un seul hidden layer de 256 neurons me parait être le meilleur compromis. Il a quatre fois 
moins de paramètres que le gros MLP de 1024 neurons cachés pour seulement 0,2 points de moins que ce dernier. 
Le MLP "bottleneck" permet d'avoir des meilleurs résultats que le linéaire (+2 points) avec un nombre de paramètres 
dans le même ordre de grandeur que la regression linéaire (11 000 vs 7 850 pour cette dernière). Ce MLP est cependant 
très instable avec SGD (83 ± 8%) : le learning rate devrait probablement être adapté.

Un point d'attention est le découpage des données pour la validation, qui peut avoir un impact important. 
D'une seed à l'autre (1337 vs 0), j'ai pu avoir un même modèle qui passe de 97,4% à 98,1% soit +0.7 points.

## Limites
- Le test n'utilise qu'seul learning rate par optimiseur
- Avec 9 époques : SGD et le momentum n'ont pas convergé sur les MLP
- un seul découpage train/validation et pas de cross validation.


## Paramètres :

```python
SEED_SPLIT = 42
VALIDATION_PROP = 0.1

EPOCHS = 9
BATCH_SIZE = 128
NB_SEEDS = 5
DEFAULT_SEED = 0


MODELES = {
    "lineaire": lineaire,
    "mlp 256": mlp_256,
    "mlp 512-256": mlp_512_256,
    "mlp botteneck" : mlp_bottleneck,
    "mlp huge" : mlp_huge,
}

# (params, pas_total) -> optimiseur
OPTIMISEURS = {
    "SGD lr 0.1": lambda p, t: SGD(p, learning_rate=0.1),
    "Momentum 0.9 lr 0.01": lambda p, t: MomentumGradient(p, learning_rate=0.01, momentum=0.9),
    "AdamW lr 1e-3": lambda p, t: AdamW(p, learning_rate=1e-3),
    # lr reduit pour compenser alpha, et schedulers sur l'entrainement
    "AdEMAMix lr 2e-4": lambda p, t: AdEMAMix(p,learning_rate=2e-4,alpha=8.0,t_alpha=t, t_beta=t),
}

```


## Meilleure epoch sur 5 seed :


| Modèle            | Optimiseur           | Acc. val           | Loss val          | Meilleure epoch |
|-------------------|----------------------|--------------------|-------------------|------------------|
| linéaire         | SGD lr 0.1           | 91.52 ± 0.18 %     | 0.3094 ± 0.0025   | 9, 9, 6, 7, 6    |
| linéaire          | Momentum 0.9 lr 0.01 | 91.55 ± 0.07 %     | 0.3069 ± 0.0009   | 5, 9, 4, 3, 5    |
| linéaire          | AdamW lr 1e-3        | 91.69 ± 0.12 %     | 0.3003 ± 0.0007   | 6, 6, 4, 8, 5    |
| linéaire          | AdEMAMix lr 2e-4     | **91.92 ± 0.06 %** | **0.2950 ± 0.0008** | 8, 7, 7, 7, 9  |
| |||||
| mlp 256           | SGD lr 0.1           | 97.50 ± 0.11 %     | 0.0846 ± 0.0027   | 9, 9, 8, 9, 7    |
| mlp 256           | Momentum 0.9 lr 0.01 | 97.51 ± 0.08 %     | 0.0852 ± 0.0034   | 8, 9, 9, 9, 9    |
| mlp 256           | AdamW lr 1e-3        | 97.67 ± 0.04 %     | **0.0799 ± 0.0022** | 6, 5, 6, 5, 6  |
| mlp 256           | AdEMAMix lr 2e-4     | **97.68 ± 0.09 %** | 0.0808 ± 0.0030   | 5, 6, 6, 6, 6    |
| |||                     |                  |
| mlp 512-256       | SGD lr 0.1           | 97.65 ± 0.09 %     | **0.0808 ± 0.0022** | 8, 5, 8, 8, 7  |
| mlp 512-256       | Momentum 0.9 lr 0.01 | 97.59 ± 0.08 %     | 0.0819 ± 0.0031   | 7, 9, 9, 8, 7    |
| mlp 512-256       | AdamW lr 1e-3        | 97.55 ± 0.23 %     | 0.0893 ± 0.0047   | 5, 2, 2, 2, 6    |
| mlp 512-256       | AdEMAMix lr 2e-4     | **97.66 ± 0.17 %** | 0.0831 ± 0.0018   | 4, 3, 3, 3, 5    |
| |||||
| mlp bottleneck    | SGD lr 0.1           | 83.30 ± 7.89 %     | 0.5299 ± 0.2378   | 9, 7, 9, 9, 9    |
| mlp bottleneck    | Momentum 0.9 lr 0.01 | 92.06 ± 0.75 %     | 0.2675 ± 0.0240   | 9, 9, 8, 9, 7    |
| mlp bottleneck    | AdamW lr 1e-3        | 93.73 ± 0.56 %     | 0.2088 ± 0.0132   | 9, 9, 9, 8, 9    |
| mlp bottleneck    | AdEMAMix lr 2e-4     | **94.09 ± 0.24 %** | **0.2004 ± 0.0072** | 9, 9, 9, 8, 9  |
| |||||
| mlp huge          | SGD lr 0.1           | 97.67 ± 0.10 %     | 0.0784 ± 0.0022   | 8, 9, 9, 9, 9    |
| mlp huge          | Momentum 0.9 lr 0.01 | 97.60 ± 0.11 %     | 0.0800 ± 0.0030   | 9, 9, 9, 9, 9    |
| mlp huge          | AdamW lr 1e-3        | 97.62 ± 0.09 %     | 0.0831 ± 0.0030   | 4, 5, 4, 4, 3    |
| mlp huge          | AdEMAMix lr 2e-4     | **97.86 ± 0.11 %** | **0.0731 ± 0.0022** | 4, 5, 4, 4, 5  |




## Details des résultats


== lineaire | SGD lr 0.1 | seed 0
  loss initiale : 2.6427  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3814 | loss val 0.3300 | acc val 90.60 %
  epoque  2/9 | loss train 0.3026 | loss val 0.3226 | acc val 91.30 %
  epoque  3/9 | loss train 0.2904 | loss val 0.3163 | acc val 91.05 %
  epoque  4/9 | loss train 0.2829 | loss val 0.3137 | acc val 91.12 %
  epoque  5/9 | loss train 0.2791 | loss val 0.3150 | acc val 91.45 %
  epoque  6/9 | loss train 0.2742 | loss val 0.3124 | acc val 91.18 %
  epoque  7/9 | loss train 0.2701 | loss val 0.3103 | acc val 91.33 %
  epoque  8/9 | loss train 0.2683 | loss val 0.3071 | acc val 91.58 %
  epoque  9/9 | loss train 0.2659 | loss val 0.3060 | acc val 91.57 %

== lineaire | SGD lr 0.1 | seed 1
  loss initiale : 2.7563  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3837 | loss val 0.3406 | acc val 90.02 %
  epoque  2/9 | loss train 0.3031 | loss val 0.3270 | acc val 90.82 %
  epoque  3/9 | loss train 0.2904 | loss val 0.3265 | acc val 90.75 %
  epoque  4/9 | loss train 0.2834 | loss val 0.3225 | acc val 90.83 %
  epoque  5/9 | loss train 0.2783 | loss val 0.3237 | acc val 90.98 %
  epoque  6/9 | loss train 0.2750 | loss val 0.3152 | acc val 91.15 %
  epoque  7/9 | loss train 0.2707 | loss val 0.3144 | acc val 91.18 %
  epoque  8/9 | loss train 0.2685 | loss val 0.3121 | acc val 91.53 %
  epoque  9/9 | loss train 0.2663 | loss val 0.3111 | acc val 91.52 %

== lineaire | SGD lr 0.1 | seed 2
  loss initiale : 2.7600  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3814 | loss val 0.3507 | acc val 90.03 %
  epoque  2/9 | loss train 0.3034 | loss val 0.3181 | acc val 91.18 %
  epoque  3/9 | loss train 0.2907 | loss val 0.3143 | acc val 91.40 %
  epoque  4/9 | loss train 0.2828 | loss val 0.3134 | acc val 91.27 %
  epoque  5/9 | loss train 0.2779 | loss val 0.3160 | acc val 91.27 %
  epoque  6/9 | loss train 0.2740 | loss val 0.3068 | acc val 91.68 %
  epoque  7/9 | loss train 0.2706 | loss val 0.3098 | acc val 91.45 %
  epoque  8/9 | loss train 0.2697 | loss val 0.3141 | acc val 91.32 %
  epoque  9/9 | loss train 0.2669 | loss val 0.3094 | acc val 91.53 %

== lineaire | SGD lr 0.1 | seed 3
  loss initiale : 2.5437  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3805 | loss val 0.3328 | acc val 90.52 %
  epoque  2/9 | loss train 0.3021 | loss val 0.3273 | acc val 90.82 %
  epoque  3/9 | loss train 0.2905 | loss val 0.3221 | acc val 90.88 %
  epoque  4/9 | loss train 0.2831 | loss val 0.3243 | acc val 90.87 %
  epoque  5/9 | loss train 0.2789 | loss val 0.3213 | acc val 91.28 %
  epoque  6/9 | loss train 0.2742 | loss val 0.3204 | acc val 91.27 %
  epoque  7/9 | loss train 0.2721 | loss val 0.3124 | acc val 91.63 %
  epoque  8/9 | loss train 0.2682 | loss val 0.3153 | acc val 91.20 %
  epoque  9/9 | loss train 0.2662 | loss val 0.3215 | acc val 91.07 %

== lineaire | SGD lr 0.1 | seed 4
  loss initiale : 2.7446  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3829 | loss val 0.3291 | acc val 90.95 %
  epoque  2/9 | loss train 0.3027 | loss val 0.3216 | acc val 91.08 %
  epoque  3/9 | loss train 0.2902 | loss val 0.3191 | acc val 91.27 %
  epoque  4/9 | loss train 0.2829 | loss val 0.3150 | acc val 91.10 %
  epoque  5/9 | loss train 0.2779 | loss val 0.3110 | acc val 91.37 %
  epoque  6/9 | loss train 0.2745 | loss val 0.3108 | acc val 91.18 %
  epoque  7/9 | loss train 0.2700 | loss val 0.3306 | acc val 90.67 %
  epoque  8/9 | loss train 0.2685 | loss val 0.3264 | acc val 90.70 %
  epoque  9/9 | loss train 0.2661 | loss val 0.3196 | acc val 91.32 %

== lineaire | Momentum 0.9 lr 0.01 | seed 0
  loss initiale : 2.6427  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3838 | loss val 0.3316 | acc val 90.82 %
  epoque  2/9 | loss train 0.2992 | loss val 0.3170 | acc val 91.10 %
  epoque  3/9 | loss train 0.2869 | loss val 0.3127 | acc val 91.35 %
  epoque  4/9 | loss train 0.2791 | loss val 0.3166 | acc val 91.03 %
  epoque  5/9 | loss train 0.2754 | loss val 0.3081 | acc val 91.50 %
  epoque  6/9 | loss train 0.2723 | loss val 0.3121 | acc val 91.25 %
  epoque  7/9 | loss train 0.2678 | loss val 0.3166 | acc val 91.28 %
  epoque  8/9 | loss train 0.2658 | loss val 0.3178 | acc val 91.20 %
  epoque  9/9 | loss train 0.2641 | loss val 0.3181 | acc val 91.27 %

== lineaire | Momentum 0.9 lr 0.01 | seed 1
  loss initiale : 2.7563  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3872 | loss val 0.3428 | acc val 90.07 %
  epoque  2/9 | loss train 0.3004 | loss val 0.3229 | acc val 90.87 %
  epoque  3/9 | loss train 0.2870 | loss val 0.3186 | acc val 91.03 %
  epoque  4/9 | loss train 0.2797 | loss val 0.3158 | acc val 91.20 %
  epoque  5/9 | loss train 0.2758 | loss val 0.3152 | acc val 90.98 %
  epoque  6/9 | loss train 0.2699 | loss val 0.3069 | acc val 91.50 %
  epoque  7/9 | loss train 0.2674 | loss val 0.3118 | acc val 91.53 %
  epoque  8/9 | loss train 0.2669 | loss val 0.3114 | acc val 91.45 %
  epoque  9/9 | loss train 0.2652 | loss val 0.3065 | acc val 91.53 %

== lineaire | Momentum 0.9 lr 0.01 | seed 2
  loss initiale : 2.7600  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3865 | loss val 0.3412 | acc val 90.32 %
  epoque  2/9 | loss train 0.3010 | loss val 0.3282 | acc val 90.80 %
  epoque  3/9 | loss train 0.2876 | loss val 0.3238 | acc val 91.07 %
  epoque  4/9 | loss train 0.2800 | loss val 0.3074 | acc val 91.52 %
  epoque  5/9 | loss train 0.2762 | loss val 0.3148 | acc val 91.07 %
  epoque  6/9 | loss train 0.2722 | loss val 0.3124 | acc val 91.33 %
  epoque  7/9 | loss train 0.2687 | loss val 0.3122 | acc val 91.08 %
  epoque  8/9 | loss train 0.2659 | loss val 0.3090 | acc val 91.60 %
  epoque  9/9 | loss train 0.2639 | loss val 0.3099 | acc val 91.53 %

== lineaire | Momentum 0.9 lr 0.01 | seed 3
  loss initiale : 2.5437  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3839 | loss val 0.3322 | acc val 90.67 %
  epoque  2/9 | loss train 0.3004 | loss val 0.3241 | acc val 91.05 %
  epoque  3/9 | loss train 0.2872 | loss val 0.3072 | acc val 91.52 %
  epoque  4/9 | loss train 0.2803 | loss val 0.3121 | acc val 91.35 %
  epoque  5/9 | loss train 0.2747 | loss val 0.3196 | acc val 91.55 %
  epoque  6/9 | loss train 0.2706 | loss val 0.3107 | acc val 91.30 %
  epoque  7/9 | loss train 0.2672 | loss val 0.3092 | acc val 91.58 %
  epoque  8/9 | loss train 0.2659 | loss val 0.3077 | acc val 91.52 %
  epoque  9/9 | loss train 0.2641 | loss val 0.3102 | acc val 91.77 %

== lineaire | Momentum 0.9 lr 0.01 | seed 4
  loss initiale : 2.7446  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3894 | loss val 0.3360 | acc val 90.53 %
  epoque  2/9 | loss train 0.2996 | loss val 0.3311 | acc val 90.35 %
  epoque  3/9 | loss train 0.2874 | loss val 0.3184 | acc val 91.25 %
  epoque  4/9 | loss train 0.2804 | loss val 0.3189 | acc val 91.30 %
  epoque  5/9 | loss train 0.2757 | loss val 0.3055 | acc val 91.68 %
  epoque  6/9 | loss train 0.2720 | loss val 0.3093 | acc val 91.38 %
  epoque  7/9 | loss train 0.2677 | loss val 0.3126 | acc val 91.53 %
  epoque  8/9 | loss train 0.2658 | loss val 0.3154 | acc val 91.08 %
  epoque  9/9 | loss train 0.2643 | loss val 0.3115 | acc val 91.60 %

== lineaire | AdamW lr 1e-3 | seed 0
  loss initiale : 2.6427  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.4669 | loss val 0.3401 | acc val 90.38 %
  epoque  2/9 | loss train 0.3033 | loss val 0.3117 | acc val 91.23 %
  epoque  3/9 | loss train 0.2854 | loss val 0.3058 | acc val 91.63 %
  epoque  4/9 | loss train 0.2757 | loss val 0.3085 | acc val 91.28 %
  epoque  5/9 | loss train 0.2710 | loss val 0.3032 | acc val 91.78 %
  epoque  6/9 | loss train 0.2677 | loss val 0.3016 | acc val 91.57 %
  epoque  7/9 | loss train 0.2631 | loss val 0.3053 | acc val 91.48 %
  epoque  8/9 | loss train 0.2603 | loss val 0.3090 | acc val 91.42 %
  epoque  9/9 | loss train 0.2583 | loss val 0.3106 | acc val 91.45 %

== lineaire | AdamW lr 1e-3 | seed 1
  loss initiale : 2.7563  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.4769 | loss val 0.3491 | acc val 89.98 %
  epoque  2/9 | loss train 0.3058 | loss val 0.3172 | acc val 91.18 %
  epoque  3/9 | loss train 0.2859 | loss val 0.3093 | acc val 91.42 %
  epoque  4/9 | loss train 0.2763 | loss val 0.3078 | acc val 91.45 %
  epoque  5/9 | loss train 0.2716 | loss val 0.3071 | acc val 91.45 %
  epoque  6/9 | loss train 0.2656 | loss val 0.2997 | acc val 91.73 %
  epoque  7/9 | loss train 0.2629 | loss val 0.3056 | acc val 91.68 %
  epoque  8/9 | loss train 0.2618 | loss val 0.3050 | acc val 91.73 %
  epoque  9/9 | loss train 0.2595 | loss val 0.3000 | acc val 91.95 %

== lineaire | AdamW lr 1e-3 | seed 2
  loss initiale : 2.7600  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.4746 | loss val 0.3458 | acc val 90.05 %
  epoque  2/9 | loss train 0.3065 | loss val 0.3205 | acc val 90.97 %
  epoque  3/9 | loss train 0.2865 | loss val 0.3212 | acc val 90.88 %
  epoque  4/9 | loss train 0.2769 | loss val 0.3002 | acc val 91.62 %
  epoque  5/9 | loss train 0.2719 | loss val 0.3051 | acc val 91.40 %
  epoque  6/9 | loss train 0.2672 | loss val 0.3027 | acc val 91.57 %
  epoque  7/9 | loss train 0.2638 | loss val 0.3036 | acc val 91.53 %
  epoque  8/9 | loss train 0.2604 | loss val 0.3005 | acc val 91.83 %
  epoque  9/9 | loss train 0.2585 | loss val 0.3020 | acc val 91.60 %

== lineaire | AdamW lr 1e-3 | seed 3
  loss initiale : 2.5437  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.4665 | loss val 0.3416 | acc val 90.48 %
  epoque  2/9 | loss train 0.3043 | loss val 0.3208 | acc val 90.80 %
  epoque  3/9 | loss train 0.2857 | loss val 0.3019 | acc val 91.78 %
  epoque  4/9 | loss train 0.2762 | loss val 0.3071 | acc val 91.60 %
  epoque  5/9 | loss train 0.2704 | loss val 0.3099 | acc val 91.58 %
  epoque  6/9 | loss train 0.2657 | loss val 0.3012 | acc val 91.57 %
  epoque  7/9 | loss train 0.2622 | loss val 0.3026 | acc val 91.82 %
  epoque  8/9 | loss train 0.2604 | loss val 0.3001 | acc val 91.63 %
  epoque  9/9 | loss train 0.2583 | loss val 0.3039 | acc val 91.82 %

== lineaire | AdamW lr 1e-3 | seed 4
  loss initiale : 2.7446  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.4765 | loss val 0.3415 | acc val 90.53 %
  epoque  2/9 | loss train 0.3044 | loss val 0.3196 | acc val 90.90 %
  epoque  3/9 | loss train 0.2861 | loss val 0.3111 | acc val 91.28 %
  epoque  4/9 | loss train 0.2765 | loss val 0.3093 | acc val 91.45 %
  epoque  5/9 | loss train 0.2711 | loss val 0.2997 | acc val 91.92 %
  epoque  6/9 | loss train 0.2672 | loss val 0.3010 | acc val 91.90 %
  epoque  7/9 | loss train 0.2628 | loss val 0.3051 | acc val 91.68 %
  epoque  8/9 | loss train 0.2605 | loss val 0.3088 | acc val 91.42 %
  epoque  9/9 | loss train 0.2587 | loss val 0.3046 | acc val 91.95 %

== lineaire | AdEMAMix lr 2e-4 | seed 0
  loss initiale : 2.6427  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.8239 | loss val 0.4290 | acc val 88.15 %
  epoque  2/9 | loss train 0.3580 | loss val 0.3397 | acc val 90.58 %
  epoque  3/9 | loss train 0.3048 | loss val 0.3130 | acc val 91.33 %
  epoque  4/9 | loss train 0.2821 | loss val 0.3035 | acc val 91.52 %
  epoque  5/9 | loss train 0.2696 | loss val 0.2986 | acc val 91.80 %
  epoque  6/9 | loss train 0.2615 | loss val 0.2974 | acc val 91.78 %
  epoque  7/9 | loss train 0.2555 | loss val 0.2954 | acc val 91.92 %
  epoque  8/9 | loss train 0.2500 | loss val 0.2939 | acc val 91.90 %
  epoque  9/9 | loss train 0.2461 | loss val 0.2963 | acc val 91.97 %

== lineaire | AdEMAMix lr 2e-4 | seed 1
  loss initiale : 2.7563  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.8435 | loss val 0.4336 | acc val 88.32 %
  epoque  2/9 | loss train 0.3641 | loss val 0.3439 | acc val 90.57 %
  epoque  3/9 | loss train 0.3069 | loss val 0.3184 | acc val 91.30 %
  epoque  4/9 | loss train 0.2832 | loss val 0.3042 | acc val 91.57 %
  epoque  5/9 | loss train 0.2705 | loss val 0.2999 | acc val 91.73 %
  epoque  6/9 | loss train 0.2617 | loss val 0.2964 | acc val 91.87 %
  epoque  7/9 | loss train 0.2557 | loss val 0.2947 | acc val 91.92 %
  epoque  8/9 | loss train 0.2507 | loss val 0.2957 | acc val 92.13 %
  epoque  9/9 | loss train 0.2464 | loss val 0.2975 | acc val 92.02 %

== lineaire | AdEMAMix lr 2e-4 | seed 2
  loss initiale : 2.7600  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.8404 | loss val 0.4371 | acc val 87.78 %
  epoque  2/9 | loss train 0.3646 | loss val 0.3469 | acc val 90.03 %
  epoque  3/9 | loss train 0.3073 | loss val 0.3225 | acc val 91.05 %
  epoque  4/9 | loss train 0.2838 | loss val 0.3061 | acc val 91.55 %
  epoque  5/9 | loss train 0.2705 | loss val 0.3029 | acc val 91.67 %
  epoque  6/9 | loss train 0.2621 | loss val 0.2985 | acc val 91.82 %
  epoque  7/9 | loss train 0.2556 | loss val 0.2946 | acc val 91.95 %
  epoque  8/9 | loss train 0.2508 | loss val 0.2954 | acc val 92.03 %
  epoque  9/9 | loss train 0.2465 | loss val 0.2986 | acc val 91.88 %

== lineaire | AdEMAMix lr 2e-4 | seed 3
  loss initiale : 2.5437  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.8080 | loss val 0.4308 | acc val 88.22 %
  epoque  2/9 | loss train 0.3588 | loss val 0.3456 | acc val 90.30 %
  epoque  3/9 | loss train 0.3046 | loss val 0.3171 | acc val 91.38 %
  epoque  4/9 | loss train 0.2822 | loss val 0.3070 | acc val 91.58 %
  epoque  5/9 | loss train 0.2697 | loss val 0.3019 | acc val 91.63 %
  epoque  6/9 | loss train 0.2614 | loss val 0.2983 | acc val 91.80 %
  epoque  7/9 | loss train 0.2554 | loss val 0.2959 | acc val 91.82 %
  epoque  8/9 | loss train 0.2500 | loss val 0.2962 | acc val 91.80 %
  epoque  9/9 | loss train 0.2468 | loss val 0.2980 | acc val 92.00 %

== lineaire | AdEMAMix lr 2e-4 | seed 4
  loss initiale : 2.7446  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.8487 | loss val 0.4365 | acc val 87.72 %
  epoque  2/9 | loss train 0.3632 | loss val 0.3427 | acc val 90.22 %
  epoque  3/9 | loss train 0.3062 | loss val 0.3179 | acc val 91.20 %
  epoque  4/9 | loss train 0.2829 | loss val 0.3063 | acc val 91.55 %
  epoque  5/9 | loss train 0.2702 | loss val 0.3005 | acc val 91.72 %
  epoque  6/9 | loss train 0.2618 | loss val 0.2992 | acc val 91.70 %
  epoque  7/9 | loss train 0.2559 | loss val 0.2968 | acc val 91.78 %
  epoque  8/9 | loss train 0.2506 | loss val 0.2974 | acc val 91.78 %
  epoque  9/9 | loss train 0.2463 | loss val 0.2957 | acc val 92.00 %

== mlp 256 | SGD lr 0.1 | seed 0
  loss initiale : 2.8183  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3048 | loss val 0.1943 | acc val 94.40 %
  epoque  2/9 | loss train 0.1473 | loss val 0.1466 | acc val 95.62 %
  epoque  3/9 | loss train 0.1065 | loss val 0.1181 | acc val 96.78 %
  epoque  4/9 | loss train 0.0826 | loss val 0.1085 | acc val 97.07 %
  epoque  5/9 | loss train 0.0672 | loss val 0.1005 | acc val 97.02 %
  epoque  6/9 | loss train 0.0556 | loss val 0.0964 | acc val 97.13 %
  epoque  7/9 | loss train 0.0464 | loss val 0.0910 | acc val 97.32 %
  epoque  8/9 | loss train 0.0389 | loss val 0.0922 | acc val 97.33 %
  epoque  9/9 | loss train 0.0334 | loss val 0.0878 | acc val 97.45 %

== mlp 256 | SGD lr 0.1 | seed 1
  loss initiale : 2.5916  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3036 | loss val 0.1859 | acc val 94.67 %
  epoque  2/9 | loss train 0.1447 | loss val 0.1362 | acc val 96.33 %
  epoque  3/9 | loss train 0.1034 | loss val 0.1149 | acc val 96.78 %
  epoque  4/9 | loss train 0.0801 | loss val 0.1004 | acc val 97.32 %
  epoque  5/9 | loss train 0.0643 | loss val 0.0901 | acc val 97.42 %
  epoque  6/9 | loss train 0.0530 | loss val 0.0869 | acc val 97.48 %
  epoque  7/9 | loss train 0.0442 | loss val 0.0866 | acc val 97.52 %
  epoque  8/9 | loss train 0.0374 | loss val 0.0860 | acc val 97.50 %
  epoque  9/9 | loss train 0.0318 | loss val 0.0812 | acc val 97.68 %

== mlp 256 | SGD lr 0.1 | seed 2
  loss initiale : 3.4346  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3036 | loss val 0.1931 | acc val 94.48 %
  epoque  2/9 | loss train 0.1429 | loss val 0.1383 | acc val 96.03 %
  epoque  3/9 | loss train 0.1023 | loss val 0.1118 | acc val 96.73 %
  epoque  4/9 | loss train 0.0798 | loss val 0.1051 | acc val 97.02 %
  epoque  5/9 | loss train 0.0647 | loss val 0.0945 | acc val 97.37 %
  epoque  6/9 | loss train 0.0533 | loss val 0.0894 | acc val 97.33 %
  epoque  7/9 | loss train 0.0448 | loss val 0.0866 | acc val 97.42 %
  epoque  8/9 | loss train 0.0380 | loss val 0.0842 | acc val 97.48 %
  epoque  9/9 | loss train 0.0323 | loss val 0.0842 | acc val 97.47 %

== mlp 256 | SGD lr 0.1 | seed 3
  loss initiale : 2.6887  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3006 | loss val 0.1959 | acc val 94.40 %
  epoque  2/9 | loss train 0.1438 | loss val 0.1362 | acc val 95.93 %
  epoque  3/9 | loss train 0.1032 | loss val 0.1207 | acc val 96.47 %
  epoque  4/9 | loss train 0.0800 | loss val 0.1010 | acc val 96.95 %
  epoque  5/9 | loss train 0.0642 | loss val 0.0965 | acc val 97.12 %
  epoque  6/9 | loss train 0.0531 | loss val 0.0928 | acc val 97.18 %
  epoque  7/9 | loss train 0.0446 | loss val 0.0862 | acc val 97.47 %
  epoque  8/9 | loss train 0.0378 | loss val 0.0863 | acc val 97.40 %
  epoque  9/9 | loss train 0.0316 | loss val 0.0823 | acc val 97.37 %

== mlp 256 | SGD lr 0.1 | seed 4
  loss initiale : 3.1283  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3058 | loss val 0.1869 | acc val 94.68 %
  epoque  2/9 | loss train 0.1471 | loss val 0.1428 | acc val 95.92 %
  epoque  3/9 | loss train 0.1058 | loss val 0.1150 | acc val 96.77 %
  epoque  4/9 | loss train 0.0815 | loss val 0.1036 | acc val 96.92 %
  epoque  5/9 | loss train 0.0654 | loss val 0.0944 | acc val 97.17 %
  epoque  6/9 | loss train 0.0544 | loss val 0.0912 | acc val 97.35 %
  epoque  7/9 | loss train 0.0450 | loss val 0.0877 | acc val 97.53 %
  epoque  8/9 | loss train 0.0381 | loss val 0.0884 | acc val 97.37 %
  epoque  9/9 | loss train 0.0325 | loss val 0.0928 | acc val 97.32 %

== mlp 256 | Momentum 0.9 lr 0.01 | seed 0
  loss initiale : 2.8183  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3210 | loss val 0.2007 | acc val 94.32 %
  epoque  2/9 | loss train 0.1501 | loss val 0.1527 | acc val 95.67 %
  epoque  3/9 | loss train 0.1089 | loss val 0.1193 | acc val 96.67 %
  epoque  4/9 | loss train 0.0856 | loss val 0.1110 | acc val 96.83 %
  epoque  5/9 | loss train 0.0687 | loss val 0.1011 | acc val 97.17 %
  epoque  6/9 | loss train 0.0569 | loss val 0.0979 | acc val 97.23 %
  epoque  7/9 | loss train 0.0479 | loss val 0.0948 | acc val 97.28 %
  epoque  8/9 | loss train 0.0403 | loss val 0.0910 | acc val 97.40 %
  epoque  9/9 | loss train 0.0344 | loss val 0.0940 | acc val 97.32 %

== mlp 256 | Momentum 0.9 lr 0.01 | seed 1
  loss initiale : 2.5916  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3201 | loss val 0.2042 | acc val 94.00 %
  epoque  2/9 | loss train 0.1489 | loss val 0.1397 | acc val 96.18 %
  epoque  3/9 | loss train 0.1065 | loss val 0.1159 | acc val 96.72 %
  epoque  4/9 | loss train 0.0818 | loss val 0.1032 | acc val 97.13 %
  epoque  5/9 | loss train 0.0659 | loss val 0.0945 | acc val 97.17 %
  epoque  6/9 | loss train 0.0541 | loss val 0.0891 | acc val 97.55 %
  epoque  7/9 | loss train 0.0451 | loss val 0.0872 | acc val 97.57 %
  epoque  8/9 | loss train 0.0387 | loss val 0.0844 | acc val 97.42 %
  epoque  9/9 | loss train 0.0322 | loss val 0.0831 | acc val 97.57 %

== mlp 256 | Momentum 0.9 lr 0.01 | seed 2
  loss initiale : 3.4346  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3259 | loss val 0.1906 | acc val 94.48 %
  epoque  2/9 | loss train 0.1437 | loss val 0.1364 | acc val 96.12 %
  epoque  3/9 | loss train 0.1043 | loss val 0.1149 | acc val 96.83 %
  epoque  4/9 | loss train 0.0812 | loss val 0.1019 | acc val 97.10 %
  epoque  5/9 | loss train 0.0652 | loss val 0.0954 | acc val 97.27 %
  epoque  6/9 | loss train 0.0543 | loss val 0.0897 | acc val 97.32 %
  epoque  7/9 | loss train 0.0454 | loss val 0.0883 | acc val 97.37 %
  epoque  8/9 | loss train 0.0385 | loss val 0.0822 | acc val 97.53 %
  epoque  9/9 | loss train 0.0326 | loss val 0.0811 | acc val 97.63 %

== mlp 256 | Momentum 0.9 lr 0.01 | seed 3
  loss initiale : 2.6887  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3210 | loss val 0.1908 | acc val 94.57 %
  epoque  2/9 | loss train 0.1461 | loss val 0.1382 | acc val 96.03 %
  epoque  3/9 | loss train 0.1051 | loss val 0.1225 | acc val 96.60 %
  epoque  4/9 | loss train 0.0809 | loss val 0.1019 | acc val 96.92 %
  epoque  5/9 | loss train 0.0647 | loss val 0.0959 | acc val 97.15 %
  epoque  6/9 | loss train 0.0536 | loss val 0.0927 | acc val 97.33 %
  epoque  7/9 | loss train 0.0457 | loss val 0.0867 | acc val 97.35 %
  epoque  8/9 | loss train 0.0378 | loss val 0.0843 | acc val 97.60 %
  epoque  9/9 | loss train 0.0320 | loss val 0.0841 | acc val 97.45 %

== mlp 256 | Momentum 0.9 lr 0.01 | seed 4
  loss initiale : 3.1283  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3270 | loss val 0.1944 | acc val 94.55 %
  epoque  2/9 | loss train 0.1479 | loss val 0.1396 | acc val 95.97 %
  epoque  3/9 | loss train 0.1066 | loss val 0.1157 | acc val 96.72 %
  epoque  4/9 | loss train 0.0826 | loss val 0.1057 | acc val 96.97 %
  epoque  5/9 | loss train 0.0663 | loss val 0.0959 | acc val 97.18 %
  epoque  6/9 | loss train 0.0552 | loss val 0.0924 | acc val 97.48 %
  epoque  7/9 | loss train 0.0459 | loss val 0.0889 | acc val 97.43 %
  epoque  8/9 | loss train 0.0392 | loss val 0.0883 | acc val 97.32 %
  epoque  9/9 | loss train 0.0330 | loss val 0.0869 | acc val 97.52 %

== mlp 256 | AdamW lr 1e-3 | seed 0
  loss initiale : 2.8183  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2592 | loss val 0.1561 | acc val 95.32 %
  epoque  2/9 | loss train 0.1064 | loss val 0.1061 | acc val 96.80 %
  epoque  3/9 | loss train 0.0702 | loss val 0.0900 | acc val 97.58 %
  epoque  4/9 | loss train 0.0514 | loss val 0.0911 | acc val 97.20 %
  epoque  5/9 | loss train 0.0376 | loss val 0.0864 | acc val 97.55 %
  epoque  6/9 | loss train 0.0289 | loss val 0.0801 | acc val 97.62 %
  epoque  7/9 | loss train 0.0231 | loss val 0.0947 | acc val 97.42 %
  epoque  8/9 | loss train 0.0184 | loss val 0.0997 | acc val 97.28 %
  epoque  9/9 | loss train 0.0148 | loss val 0.0932 | acc val 97.48 %

== mlp 256 | AdamW lr 1e-3 | seed 1
  loss initiale : 2.5916  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2556 | loss val 0.1464 | acc val 95.70 %
  epoque  2/9 | loss train 0.1046 | loss val 0.1097 | acc val 96.72 %
  epoque  3/9 | loss train 0.0703 | loss val 0.0939 | acc val 97.03 %
  epoque  4/9 | loss train 0.0478 | loss val 0.0849 | acc val 97.47 %
  epoque  5/9 | loss train 0.0362 | loss val 0.0765 | acc val 97.70 %
  epoque  6/9 | loss train 0.0277 | loss val 0.0831 | acc val 97.43 %
  epoque  7/9 | loss train 0.0206 | loss val 0.0893 | acc val 97.50 %
  epoque  8/9 | loss train 0.0177 | loss val 0.0818 | acc val 97.90 %
  epoque  9/9 | loss train 0.0157 | loss val 0.1075 | acc val 97.40 %

== mlp 256 | AdamW lr 1e-3 | seed 2
  loss initiale : 3.4346  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2673 | loss val 0.1552 | acc val 95.45 %
  epoque  2/9 | loss train 0.1076 | loss val 0.1082 | acc val 96.78 %
  epoque  3/9 | loss train 0.0692 | loss val 0.0985 | acc val 97.00 %
  epoque  4/9 | loss train 0.0508 | loss val 0.0862 | acc val 97.60 %
  epoque  5/9 | loss train 0.0374 | loss val 0.0901 | acc val 97.32 %
  epoque  6/9 | loss train 0.0295 | loss val 0.0785 | acc val 97.63 %
  epoque  7/9 | loss train 0.0220 | loss val 0.0788 | acc val 97.62 %
  epoque  8/9 | loss train 0.0184 | loss val 0.0801 | acc val 97.88 %
  epoque  9/9 | loss train 0.0144 | loss val 0.0945 | acc val 97.62 %

== mlp 256 | AdamW lr 1e-3 | seed 3
  loss initiale : 2.6887  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2595 | loss val 0.1491 | acc val 95.47 %
  epoque  2/9 | loss train 0.1048 | loss val 0.1095 | acc val 96.97 %
  epoque  3/9 | loss train 0.0691 | loss val 0.1010 | acc val 97.12 %
  epoque  4/9 | loss train 0.0489 | loss val 0.0914 | acc val 97.35 %
  epoque  5/9 | loss train 0.0360 | loss val 0.0819 | acc val 97.72 %
  epoque  6/9 | loss train 0.0271 | loss val 0.0853 | acc val 97.70 %
  epoque  7/9 | loss train 0.0220 | loss val 0.0932 | acc val 97.42 %
  epoque  8/9 | loss train 0.0179 | loss val 0.0897 | acc val 97.70 %
  epoque  9/9 | loss train 0.0138 | loss val 0.0909 | acc val 97.72 %

== mlp 256 | AdamW lr 1e-3 | seed 4
  loss initiale : 3.1283  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2726 | loss val 0.1488 | acc val 95.53 %
  epoque  2/9 | loss train 0.1079 | loss val 0.1036 | acc val 97.08 %
  epoque  3/9 | loss train 0.0712 | loss val 0.0906 | acc val 97.30 %
  epoque  4/9 | loss train 0.0512 | loss val 0.0896 | acc val 97.48 %
  epoque  5/9 | loss train 0.0382 | loss val 0.0829 | acc val 97.80 %
  epoque  6/9 | loss train 0.0287 | loss val 0.0827 | acc val 97.68 %
  epoque  7/9 | loss train 0.0223 | loss val 0.0904 | acc val 97.60 %
  epoque  8/9 | loss train 0.0172 | loss val 0.0835 | acc val 97.90 %
  epoque  9/9 | loss train 0.0158 | loss val 0.0982 | acc val 97.50 %

== mlp 256 | AdEMAMix lr 2e-4 | seed 0
  loss initiale : 2.8183  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.4398 | loss val 0.2304 | acc val 93.48 %
  epoque  2/9 | loss train 0.1672 | loss val 0.1539 | acc val 95.47 %
  epoque  3/9 | loss train 0.1044 | loss val 0.1085 | acc val 96.97 %
  epoque  4/9 | loss train 0.0674 | loss val 0.0932 | acc val 97.28 %
  epoque  5/9 | loss train 0.0420 | loss val 0.0861 | acc val 97.50 %
  epoque  6/9 | loss train 0.0254 | loss val 0.0870 | acc val 97.62 %
  epoque  7/9 | loss train 0.0139 | loss val 0.0897 | acc val 97.72 %
  epoque  8/9 | loss train 0.0070 | loss val 0.0972 | acc val 97.77 %
  epoque  9/9 | loss train 0.0032 | loss val 0.1046 | acc val 97.68 %

== mlp 256 | AdEMAMix lr 2e-4 | seed 1
  loss initiale : 2.5916  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.4305 | loss val 0.2325 | acc val 93.13 %
  epoque  2/9 | loss train 0.1662 | loss val 0.1431 | acc val 96.08 %
  epoque  3/9 | loss train 0.1012 | loss val 0.1038 | acc val 97.00 %
  epoque  4/9 | loss train 0.0628 | loss val 0.0887 | acc val 97.42 %
  epoque  5/9 | loss train 0.0391 | loss val 0.0807 | acc val 97.57 %
  epoque  6/9 | loss train 0.0227 | loss val 0.0773 | acc val 97.73 %
  epoque  7/9 | loss train 0.0126 | loss val 0.0824 | acc val 97.70 %
  epoque  8/9 | loss train 0.0059 | loss val 0.0848 | acc val 97.87 %
  epoque  9/9 | loss train 0.0025 | loss val 0.0910 | acc val 97.80 %

== mlp 256 | AdEMAMix lr 2e-4 | seed 2
  loss initiale : 3.4346  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.4681 | loss val 0.2395 | acc val 93.05 %
  epoque  2/9 | loss train 0.1713 | loss val 0.1512 | acc val 95.65 %
  epoque  3/9 | loss train 0.1062 | loss val 0.1109 | acc val 96.83 %
  epoque  4/9 | loss train 0.0676 | loss val 0.0924 | acc val 97.57 %
  epoque  5/9 | loss train 0.0427 | loss val 0.0829 | acc val 97.58 %
  epoque  6/9 | loss train 0.0249 | loss val 0.0806 | acc val 97.77 %
  epoque  7/9 | loss train 0.0136 | loss val 0.0806 | acc val 97.80 %
  epoque  8/9 | loss train 0.0065 | loss val 0.0841 | acc val 98.02 %
  epoque  9/9 | loss train 0.0028 | loss val 0.0933 | acc val 97.92 %

== mlp 256 | AdEMAMix lr 2e-4 | seed 3
  loss initiale : 2.6887  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.4397 | loss val 0.2304 | acc val 93.42 %
  epoque  2/9 | loss train 0.1690 | loss val 0.1453 | acc val 95.85 %
  epoque  3/9 | loss train 0.1025 | loss val 0.1065 | acc val 97.07 %
  epoque  4/9 | loss train 0.0641 | loss val 0.0862 | acc val 97.55 %
  epoque  5/9 | loss train 0.0400 | loss val 0.0858 | acc val 97.60 %
  epoque  6/9 | loss train 0.0234 | loss val 0.0786 | acc val 97.70 %
  epoque  7/9 | loss train 0.0129 | loss val 0.0863 | acc val 97.82 %
  epoque  8/9 | loss train 0.0061 | loss val 0.0889 | acc val 97.80 %
  epoque  9/9 | loss train 0.0026 | loss val 0.0991 | acc val 97.78 %

== mlp 256 | AdEMAMix lr 2e-4 | seed 4
  loss initiale : 3.1283  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.4687 | loss val 0.2369 | acc val 92.98 %
  epoque  2/9 | loss train 0.1720 | loss val 0.1486 | acc val 95.73 %
  epoque  3/9 | loss train 0.1065 | loss val 0.1084 | acc val 96.97 %
  epoque  4/9 | loss train 0.0674 | loss val 0.0925 | acc val 97.23 %
  epoque  5/9 | loss train 0.0422 | loss val 0.0827 | acc val 97.63 %
  epoque  6/9 | loss train 0.0254 | loss val 0.0815 | acc val 97.68 %
  epoque  7/9 | loss train 0.0137 | loss val 0.0906 | acc val 97.57 %
  epoque  8/9 | loss train 0.0069 | loss val 0.0973 | acc val 97.57 %
  epoque  9/9 | loss train 0.0031 | loss val 0.1048 | acc val 97.80 %

== mlp 512-256 | SGD lr 0.1 | seed 0
  loss initiale : 2.6520  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2743 | loss val 0.1626 | acc val 95.05 %
  epoque  2/9 | loss train 0.1122 | loss val 0.1275 | acc val 95.88 %
  epoque  3/9 | loss train 0.0734 | loss val 0.1082 | acc val 96.85 %
  epoque  4/9 | loss train 0.0519 | loss val 0.0955 | acc val 97.17 %
  epoque  5/9 | loss train 0.0378 | loss val 0.0927 | acc val 97.17 %
  epoque  6/9 | loss train 0.0274 | loss val 0.0842 | acc val 97.50 %
  epoque  7/9 | loss train 0.0199 | loss val 0.0898 | acc val 97.50 %
  epoque  8/9 | loss train 0.0149 | loss val 0.0828 | acc val 97.68 %
  epoque  9/9 | loss train 0.0111 | loss val 0.0831 | acc val 97.67 %

== mlp 512-256 | SGD lr 0.1 | seed 1
  loss initiale : 2.7504  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2739 | loss val 0.1504 | acc val 95.52 %
  epoque  2/9 | loss train 0.1096 | loss val 0.1234 | acc val 96.30 %
  epoque  3/9 | loss train 0.0732 | loss val 0.1119 | acc val 96.43 %
  epoque  4/9 | loss train 0.0506 | loss val 0.1025 | acc val 96.98 %
  epoque  5/9 | loss train 0.0361 | loss val 0.0834 | acc val 97.53 %
  epoque  6/9 | loss train 0.0255 | loss val 0.0863 | acc val 97.42 %
  epoque  7/9 | loss train 0.0186 | loss val 0.0853 | acc val 97.42 %
  epoque  8/9 | loss train 0.0135 | loss val 0.0872 | acc val 97.45 %
  epoque  9/9 | loss train 0.0101 | loss val 0.0848 | acc val 97.38 %

== mlp 512-256 | SGD lr 0.1 | seed 2
  loss initiale : 2.5397  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2748 | loss val 0.1549 | acc val 95.37 %
  epoque  2/9 | loss train 0.1091 | loss val 0.1104 | acc val 96.83 %
  epoque  3/9 | loss train 0.0724 | loss val 0.0944 | acc val 97.32 %
  epoque  4/9 | loss train 0.0516 | loss val 0.0934 | acc val 97.33 %
  epoque  5/9 | loss train 0.0368 | loss val 0.0907 | acc val 97.37 %
  epoque  6/9 | loss train 0.0270 | loss val 0.0841 | acc val 97.65 %
  epoque  7/9 | loss train 0.0199 | loss val 0.0824 | acc val 97.50 %
  epoque  8/9 | loss train 0.0144 | loss val 0.0812 | acc val 97.80 %
  epoque  9/9 | loss train 0.0106 | loss val 0.0813 | acc val 97.77 %

== mlp 512-256 | SGD lr 0.1 | seed 3
  loss initiale : 2.6232  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2760 | loss val 0.1477 | acc val 95.68 %
  epoque  2/9 | loss train 0.1119 | loss val 0.1160 | acc val 96.53 %
  epoque  3/9 | loss train 0.0725 | loss val 0.1014 | acc val 96.85 %
  epoque  4/9 | loss train 0.0505 | loss val 0.0938 | acc val 97.18 %
  epoque  5/9 | loss train 0.0359 | loss val 0.0866 | acc val 97.32 %
  epoque  6/9 | loss train 0.0257 | loss val 0.0821 | acc val 97.48 %
  epoque  7/9 | loss train 0.0178 | loss val 0.0822 | acc val 97.55 %
  epoque  8/9 | loss train 0.0132 | loss val 0.0779 | acc val 97.65 %
  epoque  9/9 | loss train 0.0095 | loss val 0.0808 | acc val 97.63 %

== mlp 512-256 | SGD lr 0.1 | seed 4
  loss initiale : 2.7285  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2777 | loss val 0.1741 | acc val 94.83 %
  epoque  2/9 | loss train 0.1133 | loss val 0.1204 | acc val 96.27 %
  epoque  3/9 | loss train 0.0749 | loss val 0.1052 | acc val 97.00 %
  epoque  4/9 | loss train 0.0529 | loss val 0.0967 | acc val 97.22 %
  epoque  5/9 | loss train 0.0369 | loss val 0.0894 | acc val 97.42 %
  epoque  6/9 | loss train 0.0273 | loss val 0.1120 | acc val 96.68 %
  epoque  7/9 | loss train 0.0198 | loss val 0.0787 | acc val 97.57 %
  epoque  8/9 | loss train 0.0144 | loss val 0.0791 | acc val 97.70 %
  epoque  9/9 | loss train 0.0108 | loss val 0.0825 | acc val 97.62 %

== mlp 512-256 | Momentum 0.9 lr 0.01 | seed 0
  loss initiale : 2.6520  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2812 | loss val 0.1504 | acc val 95.50 %
  epoque  2/9 | loss train 0.1121 | loss val 0.1198 | acc val 96.40 %
  epoque  3/9 | loss train 0.0744 | loss val 0.1124 | acc val 96.73 %
  epoque  4/9 | loss train 0.0535 | loss val 0.0935 | acc val 97.23 %
  epoque  5/9 | loss train 0.0377 | loss val 0.0925 | acc val 97.27 %
  epoque  6/9 | loss train 0.0275 | loss val 0.0897 | acc val 97.38 %
  epoque  7/9 | loss train 0.0199 | loss val 0.0852 | acc val 97.58 %
  epoque  8/9 | loss train 0.0148 | loss val 0.0855 | acc val 97.52 %
  epoque  9/9 | loss train 0.0108 | loss val 0.0863 | acc val 97.57 %

== mlp 512-256 | Momentum 0.9 lr 0.01 | seed 1
  loss initiale : 2.7504  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2859 | loss val 0.1712 | acc val 94.85 %
  epoque  2/9 | loss train 0.1116 | loss val 0.1151 | acc val 96.77 %
  epoque  3/9 | loss train 0.0745 | loss val 0.0974 | acc val 97.23 %
  epoque  4/9 | loss train 0.0515 | loss val 0.0946 | acc val 97.12 %
  epoque  5/9 | loss train 0.0365 | loss val 0.0896 | acc val 97.38 %
  epoque  6/9 | loss train 0.0260 | loss val 0.0902 | acc val 97.42 %
  epoque  7/9 | loss train 0.0192 | loss val 0.0906 | acc val 97.30 %
  epoque  8/9 | loss train 0.0135 | loss val 0.0905 | acc val 97.52 %
  epoque  9/9 | loss train 0.0101 | loss val 0.0851 | acc val 97.57 %

== mlp 512-256 | Momentum 0.9 lr 0.01 | seed 2
  loss initiale : 2.5397  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2851 | loss val 0.1537 | acc val 95.43 %
  epoque  2/9 | loss train 0.1114 | loss val 0.1209 | acc val 96.45 %
  epoque  3/9 | loss train 0.0740 | loss val 0.1028 | acc val 96.95 %
  epoque  4/9 | loss train 0.0526 | loss val 0.0895 | acc val 97.20 %
  epoque  5/9 | loss train 0.0374 | loss val 0.0981 | acc val 97.10 %
  epoque  6/9 | loss train 0.0286 | loss val 0.0892 | acc val 97.42 %
  epoque  7/9 | loss train 0.0196 | loss val 0.0828 | acc val 97.68 %
  epoque  8/9 | loss train 0.0149 | loss val 0.0837 | acc val 97.63 %
  epoque  9/9 | loss train 0.0107 | loss val 0.0789 | acc val 97.72 %

== mlp 512-256 | Momentum 0.9 lr 0.01 | seed 3
  loss initiale : 2.6232  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2851 | loss val 0.1556 | acc val 95.45 %
  epoque  2/9 | loss train 0.1136 | loss val 0.1239 | acc val 96.17 %
  epoque  3/9 | loss train 0.0734 | loss val 0.1051 | acc val 96.82 %
  epoque  4/9 | loss train 0.0524 | loss val 0.0973 | acc val 96.88 %
  epoque  5/9 | loss train 0.0371 | loss val 0.0877 | acc val 97.42 %
  epoque  6/9 | loss train 0.0271 | loss val 0.0938 | acc val 97.28 %
  epoque  7/9 | loss train 0.0192 | loss val 0.0816 | acc val 97.48 %
  epoque  8/9 | loss train 0.0140 | loss val 0.0775 | acc val 97.60 %
  epoque  9/9 | loss train 0.0099 | loss val 0.0820 | acc val 97.53 %

== mlp 512-256 | Momentum 0.9 lr 0.01 | seed 4
  loss initiale : 2.7285  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2884 | loss val 0.1560 | acc val 95.57 %
  epoque  2/9 | loss train 0.1150 | loss val 0.1158 | acc val 96.62 %
  epoque  3/9 | loss train 0.0764 | loss val 0.1045 | acc val 97.00 %
  epoque  4/9 | loss train 0.0538 | loss val 0.0952 | acc val 97.23 %
  epoque  5/9 | loss train 0.0381 | loss val 0.0875 | acc val 97.47 %
  epoque  6/9 | loss train 0.0282 | loss val 0.0877 | acc val 97.45 %
  epoque  7/9 | loss train 0.0211 | loss val 0.0828 | acc val 97.47 %
  epoque  8/9 | loss train 0.0153 | loss val 0.0849 | acc val 97.55 %
  epoque  9/9 | loss train 0.0113 | loss val 0.0841 | acc val 97.58 %

== mlp 512-256 | AdamW lr 1e-3 | seed 0
  loss initiale : 2.6520  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2095 | loss val 0.1192 | acc val 96.37 %
  epoque  2/9 | loss train 0.0837 | loss val 0.0941 | acc val 97.18 %
  epoque  3/9 | loss train 0.0519 | loss val 0.1015 | acc val 96.88 %
  epoque  4/9 | loss train 0.0377 | loss val 0.0982 | acc val 97.33 %
  epoque  5/9 | loss train 0.0331 | loss val 0.0929 | acc val 97.57 %
  epoque  6/9 | loss train 0.0222 | loss val 0.1091 | acc val 97.33 %
  epoque  7/9 | loss train 0.0234 | loss val 0.0993 | acc val 97.40 %
  epoque  8/9 | loss train 0.0221 | loss val 0.1142 | acc val 97.37 %
  epoque  9/9 | loss train 0.0210 | loss val 0.1038 | acc val 97.65 %

== mlp 512-256 | AdamW lr 1e-3 | seed 1
  loss initiale : 2.7504  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2114 | loss val 0.1327 | acc val 96.18 %
  epoque  2/9 | loss train 0.0784 | loss val 0.0873 | acc val 97.37 %
  epoque  3/9 | loss train 0.0523 | loss val 0.0900 | acc val 97.37 %
  epoque  4/9 | loss train 0.0387 | loss val 0.1025 | acc val 97.18 %
  epoque  5/9 | loss train 0.0318 | loss val 0.0990 | acc val 97.58 %
  epoque  6/9 | loss train 0.0257 | loss val 0.1548 | acc val 96.50 %
  epoque  7/9 | loss train 0.0234 | loss val 0.0979 | acc val 97.55 %
  epoque  8/9 | loss train 0.0211 | loss val 0.1192 | acc val 97.53 %
  epoque  9/9 | loss train 0.0168 | loss val 0.1110 | acc val 97.63 %

== mlp 512-256 | AdamW lr 1e-3 | seed 2
  loss initiale : 2.5397  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2113 | loss val 0.1122 | acc val 96.78 %
  epoque  2/9 | loss train 0.0827 | loss val 0.0965 | acc val 97.30 %
  epoque  3/9 | loss train 0.0513 | loss val 0.0976 | acc val 97.27 %
  epoque  4/9 | loss train 0.0377 | loss val 0.0970 | acc val 97.48 %
  epoque  5/9 | loss train 0.0315 | loss val 0.1158 | acc val 97.08 %
  epoque  6/9 | loss train 0.0254 | loss val 0.0982 | acc val 97.40 %
  epoque  7/9 | loss train 0.0210 | loss val 0.1018 | acc val 97.48 %
  epoque  8/9 | loss train 0.0204 | loss val 0.1083 | acc val 97.78 %
  epoque  9/9 | loss train 0.0181 | loss val 0.1201 | acc val 97.45 %

== mlp 512-256 | AdamW lr 1e-3 | seed 3
  loss initiale : 2.6232  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2107 | loss val 0.1140 | acc val 96.73 %
  epoque  2/9 | loss train 0.0825 | loss val 0.0855 | acc val 97.57 %
  epoque  3/9 | loss train 0.0507 | loss val 0.0924 | acc val 97.42 %
  epoque  4/9 | loss train 0.0388 | loss val 0.1072 | acc val 97.27 %
  epoque  5/9 | loss train 0.0315 | loss val 0.0874 | acc val 97.52 %
  epoque  6/9 | loss train 0.0272 | loss val 0.0963 | acc val 97.53 %
  epoque  7/9 | loss train 0.0213 | loss val 0.1251 | acc val 97.02 %
  epoque  8/9 | loss train 0.0237 | loss val 0.1208 | acc val 97.38 %
  epoque  9/9 | loss train 0.0175 | loss val 0.1094 | acc val 97.60 %

== mlp 512-256 | AdamW lr 1e-3 | seed 4
  loss initiale : 2.7285  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2128 | loss val 0.1233 | acc val 96.35 %
  epoque  2/9 | loss train 0.0822 | loss val 0.0902 | acc val 97.27 %
  epoque  3/9 | loss train 0.0521 | loss val 0.0852 | acc val 97.43 %
  epoque  4/9 | loss train 0.0386 | loss val 0.0997 | acc val 97.38 %
  epoque  5/9 | loss train 0.0293 | loss val 0.0922 | acc val 97.43 %
  epoque  6/9 | loss train 0.0262 | loss val 0.0842 | acc val 97.97 %
  epoque  7/9 | loss train 0.0219 | loss val 0.1069 | acc val 97.35 %
  epoque  8/9 | loss train 0.0227 | loss val 0.1025 | acc val 97.58 %
  epoque  9/9 | loss train 0.0210 | loss val 0.1044 | acc val 97.57 %

== mlp 512-256 | AdEMAMix lr 2e-4 | seed 0
  loss initiale : 2.6520  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3141 | loss val 0.1441 | acc val 95.77 %
  epoque  2/9 | loss train 0.0998 | loss val 0.0981 | acc val 97.13 %
  epoque  3/9 | loss train 0.0524 | loss val 0.0939 | acc val 97.23 %
  epoque  4/9 | loss train 0.0267 | loss val 0.0829 | acc val 97.65 %
  epoque  5/9 | loss train 0.0131 | loss val 0.0898 | acc val 97.83 %
  epoque  6/9 | loss train 0.0080 | loss val 0.1132 | acc val 97.63 %
  epoque  7/9 | loss train 0.0122 | loss val 0.1252 | acc val 97.48 %
  epoque  8/9 | loss train 0.0127 | loss val 0.1087 | acc val 97.85 %
  epoque  9/9 | loss train 0.0135 | loss val 0.1318 | acc val 97.70 %

== mlp 512-256 | AdEMAMix lr 2e-4 | seed 1
  loss initiale : 2.7504  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3194 | loss val 0.1659 | acc val 94.98 %
  epoque  2/9 | loss train 0.0996 | loss val 0.0973 | acc val 97.20 %
  epoque  3/9 | loss train 0.0503 | loss val 0.0847 | acc val 97.63 %
  epoque  4/9 | loss train 0.0258 | loss val 0.0873 | acc val 97.75 %
  epoque  5/9 | loss train 0.0119 | loss val 0.0910 | acc val 97.88 %
  epoque  6/9 | loss train 0.0075 | loss val 0.1084 | acc val 97.63 %
  epoque  7/9 | loss train 0.0091 | loss val 0.1327 | acc val 97.30 %
  epoque  8/9 | loss train 0.0124 | loss val 0.1297 | acc val 97.55 %
  epoque  9/9 | loss train 0.0145 | loss val 0.1407 | acc val 97.42 %

== mlp 512-256 | AdEMAMix lr 2e-4 | seed 2
  loss initiale : 2.5397  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3163 | loss val 0.1507 | acc val 95.60 %
  epoque  2/9 | loss train 0.0998 | loss val 0.1046 | acc val 97.10 %
  epoque  3/9 | loss train 0.0517 | loss val 0.0798 | acc val 97.77 %
  epoque  4/9 | loss train 0.0263 | loss val 0.0834 | acc val 97.77 %
  epoque  5/9 | loss train 0.0130 | loss val 0.0856 | acc val 97.93 %
  epoque  6/9 | loss train 0.0081 | loss val 0.1019 | acc val 97.72 %
  epoque  7/9 | loss train 0.0094 | loss val 0.1274 | acc val 97.53 %
  epoque  8/9 | loss train 0.0127 | loss val 0.1329 | acc val 97.53 %
  epoque  9/9 | loss train 0.0126 | loss val 0.1533 | acc val 97.27 %

== mlp 512-256 | AdEMAMix lr 2e-4 | seed 3
  loss initiale : 2.6232  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3173 | loss val 0.1492 | acc val 95.97 %
  epoque  2/9 | loss train 0.1018 | loss val 0.1068 | acc val 96.80 %
  epoque  3/9 | loss train 0.0491 | loss val 0.0839 | acc val 97.37 %
  epoque  4/9 | loss train 0.0254 | loss val 0.0921 | acc val 97.55 %
  epoque  5/9 | loss train 0.0127 | loss val 0.0922 | acc val 97.83 %
  epoque  6/9 | loss train 0.0080 | loss val 0.0990 | acc val 97.87 %
  epoque  7/9 | loss train 0.0097 | loss val 0.1299 | acc val 97.62 %
  epoque  8/9 | loss train 0.0138 | loss val 0.1358 | acc val 97.55 %
  epoque  9/9 | loss train 0.0146 | loss val 0.1404 | acc val 97.50 %

== mlp 512-256 | AdEMAMix lr 2e-4 | seed 4
  loss initiale : 2.7285  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3212 | loss val 0.1520 | acc val 95.72 %
  epoque  2/9 | loss train 0.1019 | loss val 0.0980 | acc val 96.97 %
  epoque  3/9 | loss train 0.0520 | loss val 0.0929 | acc val 97.20 %
  epoque  4/9 | loss train 0.0255 | loss val 0.0884 | acc val 97.53 %
  epoque  5/9 | loss train 0.0130 | loss val 0.0841 | acc val 97.88 %
  epoque  6/9 | loss train 0.0078 | loss val 0.1017 | acc val 97.78 %
  epoque  7/9 | loss train 0.0094 | loss val 0.1292 | acc val 97.57 %
  epoque  8/9 | loss train 0.0130 | loss val 0.1126 | acc val 97.77 %
  epoque  9/9 | loss train 0.0156 | loss val 0.1262 | acc val 97.57 %

== mlp botteneck | SGD lr 0.1 | seed 0
  loss initiale : 2.5805  (log(10) = 2.3026)
  epoque  1/9 | loss train 1.8362 | loss val 1.5671 | acc val 40.87 %
  epoque  2/9 | loss train 1.5373 | loss val 1.2356 | acc val 55.25 %
  epoque  3/9 | loss train 1.5014 | loss val 1.3691 | acc val 46.13 %
  epoque  4/9 | loss train 1.2721 | loss val 0.9837 | acc val 64.10 %
  epoque  5/9 | loss train 1.0613 | loss val 1.1655 | acc val 57.82 %
  epoque  6/9 | loss train 0.9975 | loss val 2.9641 | acc val 52.62 %
  epoque  7/9 | loss train 1.0013 | loss val 0.8061 | acc val 71.55 %
  epoque  8/9 | loss train 0.7826 | loss val 0.7310 | acc val 75.55 %
  epoque  9/9 | loss train 0.7159 | loss val 0.6903 | acc val 77.72 %

== mlp botteneck | SGD lr 0.1 | seed 1
  loss initiale : 2.5622  (log(10) = 2.3026)
  epoque  1/9 | loss train 1.2024 | loss val 1.1078 | acc val 60.20 %
  epoque  2/9 | loss train 0.9222 | loss val 0.5175 | acc val 85.82 %
  epoque  3/9 | loss train 0.5115 | loss val 0.4818 | acc val 86.82 %
  epoque  4/9 | loss train 0.4455 | loss val 0.4054 | acc val 88.03 %
  epoque  5/9 | loss train 0.3842 | loss val 0.3762 | acc val 89.00 %
  epoque  6/9 | loss train 0.3636 | loss val 0.3702 | acc val 88.90 %
  epoque  7/9 | loss train 0.3275 | loss val 0.3369 | acc val 89.78 %
  epoque  8/9 | loss train 0.3113 | loss val 0.3389 | acc val 89.72 %
  epoque  9/9 | loss train 0.3041 | loss val 0.3408 | acc val 90.08 %

== mlp botteneck | SGD lr 0.1 | seed 2
  loss initiale : 2.7075  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.8861 | loss val 0.4643 | acc val 86.87 %
  epoque  2/9 | loss train 0.4416 | loss val 0.3852 | acc val 88.88 %
  epoque  3/9 | loss train 0.4134 | loss val 0.4300 | acc val 88.53 %
  epoque  4/9 | loss train 0.3587 | loss val 0.3699 | acc val 89.48 %
  epoque  5/9 | loss train 0.3402 | loss val 0.4578 | acc val 87.18 %
  epoque  6/9 | loss train 0.3200 | loss val 0.3585 | acc val 90.07 %
  epoque  7/9 | loss train 0.3013 | loss val 0.3188 | acc val 90.78 %
  epoque  8/9 | loss train 0.2863 | loss val 0.3855 | acc val 89.25 %
  epoque  9/9 | loss train 0.2815 | loss val 0.2965 | acc val 91.28 %

== mlp botteneck | SGD lr 0.1 | seed 3
  loss initiale : 2.6668  (log(10) = 2.3026)
  epoque  1/9 | loss train 1.1966 | loss val 1.3583 | acc val 51.12 %
  epoque  2/9 | loss train 0.9005 | loss val 0.8916 | acc val 72.32 %
  epoque  3/9 | loss train 0.7064 | loss val 0.6598 | acc val 80.18 %
  epoque  4/9 | loss train 0.5284 | loss val 0.6886 | acc val 79.15 %
  epoque  5/9 | loss train 0.4529 | loss val 0.5647 | acc val 81.22 %
  epoque  6/9 | loss train 0.4293 | loss val 0.4964 | acc val 83.82 %
  epoque  7/9 | loss train 0.4091 | loss val 0.4400 | acc val 86.23 %
  epoque  8/9 | loss train 0.3875 | loss val 0.4158 | acc val 87.20 %
  epoque  9/9 | loss train 0.3735 | loss val 0.4080 | acc val 87.08 %

== mlp botteneck | SGD lr 0.1 | seed 4
  loss initiale : 2.7303  (log(10) = 2.3026)
  epoque  1/9 | loss train 1.5643 | loss val 1.4946 | acc val 40.48 %
  epoque  2/9 | loss train 1.5725 | loss val 1.3263 | acc val 49.07 %
  epoque  3/9 | loss train 1.3530 | loss val 1.9179 | acc val 31.12 %
  epoque  4/9 | loss train 1.5100 | loss val 1.8932 | acc val 34.78 %
  epoque  5/9 | loss train 1.4479 | loss val 1.3054 | acc val 49.33 %
  epoque  6/9 | loss train 1.3697 | loss val 1.2794 | acc val 49.72 %
  epoque  7/9 | loss train 1.2681 | loss val 1.1271 | acc val 60.67 %
  epoque  8/9 | loss train 1.0073 | loss val 0.9208 | acc val 68.37 %
  epoque  9/9 | loss train 0.8994 | loss val 0.9177 | acc val 70.65 %

== mlp botteneck | Momentum 0.9 lr 0.01 | seed 0
  loss initiale : 2.5805  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.6090 | loss val 0.4689 | acc val 86.22 %
  epoque  2/9 | loss train 0.4040 | loss val 0.3914 | acc val 88.60 %
  epoque  3/9 | loss train 0.3615 | loss val 0.3838 | acc val 89.42 %
  epoque  4/9 | loss train 0.3437 | loss val 0.3759 | acc val 89.20 %
  epoque  5/9 | loss train 0.3457 | loss val 0.4059 | acc val 88.20 %
  epoque  6/9 | loss train 0.3163 | loss val 0.3636 | acc val 89.15 %
  epoque  7/9 | loss train 0.3125 | loss val 0.3309 | acc val 90.60 %
  epoque  8/9 | loss train 0.2824 | loss val 0.3173 | acc val 91.13 %
  epoque  9/9 | loss train 0.2771 | loss val 0.3061 | acc val 90.77 %

== mlp botteneck | Momentum 0.9 lr 0.01 | seed 1
  loss initiale : 2.5622  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.5038 | loss val 0.3397 | acc val 89.63 %
  epoque  2/9 | loss train 0.3243 | loss val 0.3216 | acc val 90.67 %
  epoque  3/9 | loss train 0.2905 | loss val 0.3054 | acc val 91.12 %
  epoque  4/9 | loss train 0.2614 | loss val 0.2920 | acc val 91.37 %
  epoque  5/9 | loss train 0.2488 | loss val 0.2705 | acc val 91.93 %
  epoque  6/9 | loss train 0.2341 | loss val 0.2508 | acc val 92.80 %
  epoque  7/9 | loss train 0.2192 | loss val 0.2860 | acc val 91.92 %
  epoque  8/9 | loss train 0.2103 | loss val 0.2588 | acc val 92.65 %
  epoque  9/9 | loss train 0.2053 | loss val 0.2410 | acc val 92.73 %

== mlp botteneck | Momentum 0.9 lr 0.01 | seed 2
  loss initiale : 2.7075  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.6739 | loss val 0.5082 | acc val 84.97 %
  epoque  2/9 | loss train 0.4495 | loss val 0.4277 | acc val 87.53 %
  epoque  3/9 | loss train 0.3742 | loss val 0.4067 | acc val 88.27 %
  epoque  4/9 | loss train 0.3321 | loss val 0.3335 | acc val 90.17 %
  epoque  5/9 | loss train 0.2989 | loss val 0.3405 | acc val 90.35 %
  epoque  6/9 | loss train 0.2853 | loss val 0.3087 | acc val 90.93 %
  epoque  7/9 | loss train 0.2664 | loss val 0.3548 | acc val 89.92 %
  epoque  8/9 | loss train 0.2604 | loss val 0.2839 | acc val 91.68 %
  epoque  9/9 | loss train 0.2457 | loss val 0.2971 | acc val 91.65 %

== mlp botteneck | Momentum 0.9 lr 0.01 | seed 3
  loss initiale : 2.6668  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.6930 | loss val 0.5134 | acc val 85.37 %
  epoque  2/9 | loss train 0.4360 | loss val 0.5046 | acc val 85.83 %
  epoque  3/9 | loss train 0.3910 | loss val 0.4121 | acc val 88.25 %
  epoque  4/9 | loss train 0.3566 | loss val 0.3569 | acc val 89.53 %
  epoque  5/9 | loss train 0.3168 | loss val 0.3379 | acc val 90.37 %
  epoque  6/9 | loss train 0.2923 | loss val 0.3253 | acc val 91.17 %
  epoque  7/9 | loss train 0.2574 | loss val 0.2823 | acc val 91.77 %
  epoque  8/9 | loss train 0.2448 | loss val 0.2828 | acc val 91.67 %
  epoque  9/9 | loss train 0.2220 | loss val 0.2544 | acc val 92.57 %

== mlp botteneck | Momentum 0.9 lr 0.01 | seed 4
  loss initiale : 2.7303  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.6232 | loss val 0.5108 | acc val 85.50 %
  epoque  2/9 | loss train 0.3984 | loss val 0.3649 | acc val 89.55 %
  epoque  3/9 | loss train 0.3324 | loss val 0.3276 | acc val 90.83 %
  epoque  4/9 | loss train 0.2923 | loss val 0.3340 | acc val 90.58 %
  epoque  5/9 | loss train 0.2792 | loss val 0.3109 | acc val 90.98 %
  epoque  6/9 | loss train 0.2530 | loss val 0.2735 | acc val 92.37 %
  epoque  7/9 | loss train 0.2361 | loss val 0.2520 | acc val 92.57 %
  epoque  8/9 | loss train 0.2150 | loss val 0.2901 | acc val 92.00 %
  epoque  9/9 | loss train 0.2161 | loss val 0.2649 | acc val 92.68 %

== mlp botteneck | AdamW lr 1e-3 | seed 0
  loss initiale : 2.5805  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.5347 | loss val 0.3402 | acc val 89.58 %
  epoque  2/9 | loss train 0.2928 | loss val 0.2819 | acc val 91.07 %
  epoque  3/9 | loss train 0.2557 | loss val 0.2643 | acc val 91.75 %
  epoque  4/9 | loss train 0.2394 | loss val 0.2767 | acc val 91.40 %
  epoque  5/9 | loss train 0.2248 | loss val 0.2587 | acc val 92.08 %
  epoque  6/9 | loss train 0.2164 | loss val 0.2654 | acc val 91.75 %
  epoque  7/9 | loss train 0.2081 | loss val 0.2404 | acc val 92.88 %
  epoque  8/9 | loss train 0.2010 | loss val 0.2645 | acc val 92.05 %
  epoque  9/9 | loss train 0.1994 | loss val 0.2352 | acc val 92.73 %

== mlp botteneck | AdamW lr 1e-3 | seed 1
  loss initiale : 2.5622  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.5117 | loss val 0.3141 | acc val 90.52 %
  epoque  2/9 | loss train 0.2714 | loss val 0.2554 | acc val 92.38 %
  epoque  3/9 | loss train 0.2309 | loss val 0.2446 | acc val 92.58 %
  epoque  4/9 | loss train 0.2102 | loss val 0.2247 | acc val 92.98 %
  epoque  5/9 | loss train 0.1973 | loss val 0.2262 | acc val 93.05 %
  epoque  6/9 | loss train 0.1872 | loss val 0.2133 | acc val 93.32 %
  epoque  7/9 | loss train 0.1804 | loss val 0.2111 | acc val 93.40 %
  epoque  8/9 | loss train 0.1725 | loss val 0.2085 | acc val 93.62 %
  epoque  9/9 | loss train 0.1686 | loss val 0.2012 | acc val 93.78 %

== mlp botteneck | AdamW lr 1e-3 | seed 2
  loss initiale : 2.7075  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.5615 | loss val 0.3208 | acc val 90.37 %
  epoque  2/9 | loss train 0.2730 | loss val 0.2544 | acc val 92.27 %
  epoque  3/9 | loss train 0.2332 | loss val 0.2468 | acc val 92.67 %
  epoque  4/9 | loss train 0.2131 | loss val 0.2181 | acc val 93.23 %
  epoque  5/9 | loss train 0.1954 | loss val 0.2232 | acc val 93.47 %
  epoque  6/9 | loss train 0.1888 | loss val 0.2151 | acc val 93.77 %
  epoque  7/9 | loss train 0.1769 | loss val 0.2323 | acc val 92.77 %
  epoque  8/9 | loss train 0.1748 | loss val 0.2154 | acc val 93.70 %
  epoque  9/9 | loss train 0.1674 | loss val 0.2028 | acc val 93.88 %

== mlp botteneck | AdamW lr 1e-3 | seed 3
  loss initiale : 2.6668  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.6216 | loss val 0.3586 | acc val 89.08 %
  epoque  2/9 | loss train 0.2871 | loss val 0.2707 | acc val 91.60 %
  epoque  3/9 | loss train 0.2325 | loss val 0.2401 | acc val 92.63 %
  epoque  4/9 | loss train 0.2117 | loss val 0.2190 | acc val 93.28 %
  epoque  5/9 | loss train 0.1948 | loss val 0.2224 | acc val 93.43 %
  epoque  6/9 | loss train 0.1850 | loss val 0.2194 | acc val 93.47 %
  epoque  7/9 | loss train 0.1760 | loss val 0.2200 | acc val 93.32 %
  epoque  8/9 | loss train 0.1688 | loss val 0.2025 | acc val 93.82 %
  epoque  9/9 | loss train 0.1637 | loss val 0.2204 | acc val 93.35 %

== mlp botteneck | AdamW lr 1e-3 | seed 4
  loss initiale : 2.7303  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.5731 | loss val 0.3321 | acc val 90.20 %
  epoque  2/9 | loss train 0.2814 | loss val 0.2825 | acc val 91.35 %
  epoque  3/9 | loss train 0.2295 | loss val 0.2394 | acc val 92.67 %
  epoque  4/9 | loss train 0.2074 | loss val 0.2399 | acc val 93.25 %
  epoque  5/9 | loss train 0.1931 | loss val 0.2204 | acc val 93.47 %
  epoque  6/9 | loss train 0.1826 | loss val 0.2075 | acc val 93.68 %
  epoque  7/9 | loss train 0.1763 | loss val 0.2144 | acc val 93.82 %
  epoque  8/9 | loss train 0.1701 | loss val 0.2204 | acc val 93.55 %
  epoque  9/9 | loss train 0.1661 | loss val 0.2021 | acc val 94.45 %

== mlp botteneck | AdEMAMix lr 2e-4 | seed 0
  loss initiale : 2.5805  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.9125 | loss val 0.4136 | acc val 87.33 %
  epoque  2/9 | loss train 0.3467 | loss val 0.3198 | acc val 90.13 %
  epoque  3/9 | loss train 0.2843 | loss val 0.2872 | acc val 91.12 %
  epoque  4/9 | loss train 0.2485 | loss val 0.2546 | acc val 92.07 %
  epoque  5/9 | loss train 0.2235 | loss val 0.2458 | acc val 92.50 %
  epoque  6/9 | loss train 0.2073 | loss val 0.2348 | acc val 92.92 %
  epoque  7/9 | loss train 0.1926 | loss val 0.2155 | acc val 93.23 %
  epoque  8/9 | loss train 0.1773 | loss val 0.2071 | acc val 93.78 %
  epoque  9/9 | loss train 0.1668 | loss val 0.1998 | acc val 93.83 %

== mlp botteneck | AdEMAMix lr 2e-4 | seed 1
  loss initiale : 2.5622  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.8720 | loss val 0.4146 | acc val 88.05 %
  epoque  2/9 | loss train 0.3303 | loss val 0.3077 | acc val 90.95 %
  epoque  3/9 | loss train 0.2618 | loss val 0.2647 | acc val 91.90 %
  epoque  4/9 | loss train 0.2230 | loss val 0.2313 | acc val 93.05 %
  epoque  5/9 | loss train 0.1970 | loss val 0.2200 | acc val 93.52 %
  epoque  6/9 | loss train 0.1799 | loss val 0.2040 | acc val 94.03 %
  epoque  7/9 | loss train 0.1662 | loss val 0.1991 | acc val 93.95 %
  epoque  8/9 | loss train 0.1558 | loss val 0.1905 | acc val 94.30 %
  epoque  9/9 | loss train 0.1469 | loss val 0.1898 | acc val 94.47 %

== mlp botteneck | AdEMAMix lr 2e-4 | seed 2
  loss initiale : 2.7075  (log(10) = 2.3026)
  epoque  1/9 | loss train 1.0007 | loss val 0.4770 | acc val 85.15 %
  epoque  2/9 | loss train 0.3703 | loss val 0.3209 | acc val 89.72 %
  epoque  3/9 | loss train 0.2796 | loss val 0.2664 | acc val 91.85 %
  epoque  4/9 | loss train 0.2395 | loss val 0.2439 | acc val 92.58 %
  epoque  5/9 | loss train 0.2138 | loss val 0.2336 | acc val 93.08 %
  epoque  6/9 | loss train 0.1953 | loss val 0.2172 | acc val 93.13 %
  epoque  7/9 | loss train 0.1790 | loss val 0.2159 | acc val 93.52 %
  epoque  8/9 | loss train 0.1669 | loss val 0.2056 | acc val 93.90 %
  epoque  9/9 | loss train 0.1559 | loss val 0.2022 | acc val 93.85 %

== mlp botteneck | AdEMAMix lr 2e-4 | seed 3
  loss initiale : 2.6668  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.9720 | loss val 0.4572 | acc val 86.12 %
  epoque  2/9 | loss train 0.3615 | loss val 0.3424 | acc val 89.78 %
  epoque  3/9 | loss train 0.2845 | loss val 0.2838 | acc val 91.47 %
  epoque  4/9 | loss train 0.2416 | loss val 0.2579 | acc val 92.30 %
  epoque  5/9 | loss train 0.2137 | loss val 0.2381 | acc val 92.95 %
  epoque  6/9 | loss train 0.1936 | loss val 0.2285 | acc val 92.93 %
  epoque  7/9 | loss train 0.1770 | loss val 0.2200 | acc val 93.68 %
  epoque  8/9 | loss train 0.1643 | loss val 0.2120 | acc val 94.08 %
  epoque  9/9 | loss train 0.1551 | loss val 0.2123 | acc val 93.95 %

== mlp botteneck | AdEMAMix lr 2e-4 | seed 4
  loss initiale : 2.7303  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.9134 | loss val 0.4098 | acc val 87.60 %
  epoque  2/9 | loss train 0.3322 | loss val 0.3022 | acc val 90.63 %
  epoque  3/9 | loss train 0.2591 | loss val 0.2621 | acc val 92.05 %
  epoque  4/9 | loss train 0.2218 | loss val 0.2343 | acc val 92.77 %
  epoque  5/9 | loss train 0.1964 | loss val 0.2164 | acc val 93.72 %
  epoque  6/9 | loss train 0.1807 | loss val 0.2104 | acc val 93.83 %
  epoque  7/9 | loss train 0.1683 | loss val 0.2064 | acc val 93.82 %
  epoque  8/9 | loss train 0.1577 | loss val 0.2039 | acc val 94.32 %
  epoque  9/9 | loss train 0.1490 | loss val 0.1982 | acc val 94.22 %

== mlp huge | SGD lr 0.1 | seed 0
  loss initiale : 2.8207  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2907 | loss val 0.1704 | acc val 95.08 %
  epoque  2/9 | loss train 0.1233 | loss val 0.1270 | acc val 96.20 %
  epoque  3/9 | loss train 0.0853 | loss val 0.1093 | acc val 96.80 %
  epoque  4/9 | loss train 0.0644 | loss val 0.0966 | acc val 97.23 %
  epoque  5/9 | loss train 0.0492 | loss val 0.0915 | acc val 97.37 %
  epoque  6/9 | loss train 0.0396 | loss val 0.0871 | acc val 97.45 %
  epoque  7/9 | loss train 0.0317 | loss val 0.0899 | acc val 97.42 %
  epoque  8/9 | loss train 0.0255 | loss val 0.0824 | acc val 97.57 %
  epoque  9/9 | loss train 0.0207 | loss val 0.0911 | acc val 97.43 %

== mlp huge | SGD lr 0.1 | seed 1
  loss initiale : 2.6559  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2998 | loss val 0.1654 | acc val 95.17 %
  epoque  2/9 | loss train 0.1231 | loss val 0.1234 | acc val 96.38 %
  epoque  3/9 | loss train 0.0848 | loss val 0.1071 | acc val 96.85 %
  epoque  4/9 | loss train 0.0634 | loss val 0.0971 | acc val 97.10 %
  epoque  5/9 | loss train 0.0493 | loss val 0.0903 | acc val 97.35 %
  epoque  6/9 | loss train 0.0387 | loss val 0.0831 | acc val 97.45 %
  epoque  7/9 | loss train 0.0312 | loss val 0.0801 | acc val 97.52 %
  epoque  8/9 | loss train 0.0253 | loss val 0.0787 | acc val 97.47 %
  epoque  9/9 | loss train 0.0208 | loss val 0.0771 | acc val 97.58 %

== mlp huge | SGD lr 0.1 | seed 2
  loss initiale : 2.8004  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2891 | loss val 0.1647 | acc val 95.30 %
  epoque  2/9 | loss train 0.1220 | loss val 0.1248 | acc val 96.37 %
  epoque  3/9 | loss train 0.0843 | loss val 0.1093 | acc val 96.85 %
  epoque  4/9 | loss train 0.0630 | loss val 0.0902 | acc val 97.53 %
  epoque  5/9 | loss train 0.0481 | loss val 0.0885 | acc val 97.42 %
  epoque  6/9 | loss train 0.0385 | loss val 0.0824 | acc val 97.72 %
  epoque  7/9 | loss train 0.0309 | loss val 0.0862 | acc val 97.47 %
  epoque  8/9 | loss train 0.0251 | loss val 0.0804 | acc val 97.78 %
  epoque  9/9 | loss train 0.0204 | loss val 0.0766 | acc val 97.75 %

== mlp huge | SGD lr 0.1 | seed 3
  loss initiale : 2.5153  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2920 | loss val 0.1707 | acc val 95.02 %
  epoque  2/9 | loss train 0.1223 | loss val 0.1272 | acc val 96.38 %
  epoque  3/9 | loss train 0.0852 | loss val 0.1060 | acc val 96.78 %
  epoque  4/9 | loss train 0.0636 | loss val 0.0962 | acc val 96.97 %
  epoque  5/9 | loss train 0.0489 | loss val 0.0884 | acc val 97.38 %
  epoque  6/9 | loss train 0.0386 | loss val 0.0837 | acc val 97.48 %
  epoque  7/9 | loss train 0.0309 | loss val 0.0865 | acc val 97.40 %
  epoque  8/9 | loss train 0.0247 | loss val 0.0813 | acc val 97.55 %
  epoque  9/9 | loss train 0.0206 | loss val 0.0767 | acc val 97.82 %

== mlp huge | SGD lr 0.1 | seed 4
  loss initiale : 2.5715  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2890 | loss val 0.1651 | acc val 95.58 %
  epoque  2/9 | loss train 0.1249 | loss val 0.1239 | acc val 96.47 %
  epoque  3/9 | loss train 0.0875 | loss val 0.1103 | acc val 96.82 %
  epoque  4/9 | loss train 0.0647 | loss val 0.1021 | acc val 96.88 %
  epoque  5/9 | loss train 0.0502 | loss val 0.0896 | acc val 97.30 %
  epoque  6/9 | loss train 0.0395 | loss val 0.0864 | acc val 97.38 %
  epoque  7/9 | loss train 0.0314 | loss val 0.0839 | acc val 97.57 %
  epoque  8/9 | loss train 0.0255 | loss val 0.0875 | acc val 97.48 %
  epoque  9/9 | loss train 0.0210 | loss val 0.0790 | acc val 97.63 %

== mlp huge | Momentum 0.9 lr 0.01 | seed 0
  loss initiale : 2.8207  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2860 | loss val 0.1791 | acc val 94.62 %
  epoque  2/9 | loss train 0.1279 | loss val 0.1330 | acc val 96.20 %
  epoque  3/9 | loss train 0.0882 | loss val 0.1080 | acc val 97.03 %
  epoque  4/9 | loss train 0.0656 | loss val 0.1021 | acc val 97.07 %
  epoque  5/9 | loss train 0.0504 | loss val 0.0944 | acc val 97.18 %
  epoque  6/9 | loss train 0.0400 | loss val 0.0892 | acc val 97.23 %
  epoque  7/9 | loss train 0.0322 | loss val 0.0854 | acc val 97.50 %
  epoque  8/9 | loss train 0.0257 | loss val 0.0857 | acc val 97.58 %
  epoque  9/9 | loss train 0.0211 | loss val 0.0845 | acc val 97.62 %

== mlp huge | Momentum 0.9 lr 0.01 | seed 1
  loss initiale : 2.6559  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2838 | loss val 0.1769 | acc val 94.87 %
  epoque  2/9 | loss train 0.1283 | loss val 0.1237 | acc val 96.45 %
  epoque  3/9 | loss train 0.0872 | loss val 0.1057 | acc val 96.92 %
  epoque  4/9 | loss train 0.0653 | loss val 0.1002 | acc val 96.98 %
  epoque  5/9 | loss train 0.0507 | loss val 0.0878 | acc val 97.33 %
  epoque  6/9 | loss train 0.0394 | loss val 0.0844 | acc val 97.45 %
  epoque  7/9 | loss train 0.0318 | loss val 0.0805 | acc val 97.55 %
  epoque  8/9 | loss train 0.0256 | loss val 0.0792 | acc val 97.57 %
  epoque  9/9 | loss train 0.0209 | loss val 0.0786 | acc val 97.45 %

== mlp huge | Momentum 0.9 lr 0.01 | seed 2
  loss initiale : 2.8004  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2809 | loss val 0.1682 | acc val 95.25 %
  epoque  2/9 | loss train 0.1254 | loss val 0.1233 | acc val 96.58 %
  epoque  3/9 | loss train 0.0871 | loss val 0.1070 | acc val 97.05 %
  epoque  4/9 | loss train 0.0647 | loss val 0.0908 | acc val 97.52 %
  epoque  5/9 | loss train 0.0498 | loss val 0.0919 | acc val 97.27 %
  epoque  6/9 | loss train 0.0392 | loss val 0.0829 | acc val 97.60 %
  epoque  7/9 | loss train 0.0313 | loss val 0.0857 | acc val 97.50 %
  epoque  8/9 | loss train 0.0256 | loss val 0.0802 | acc val 97.73 %
  epoque  9/9 | loss train 0.0210 | loss val 0.0773 | acc val 97.77 %

== mlp huge | Momentum 0.9 lr 0.01 | seed 3
  loss initiale : 2.5153  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2830 | loss val 0.1711 | acc val 94.97 %
  epoque  2/9 | loss train 0.1266 | loss val 0.1279 | acc val 96.32 %
  epoque  3/9 | loss train 0.0875 | loss val 0.1121 | acc val 96.68 %
  epoque  4/9 | loss train 0.0645 | loss val 0.0963 | acc val 96.98 %
  epoque  5/9 | loss train 0.0502 | loss val 0.0910 | acc val 97.22 %
  epoque  6/9 | loss train 0.0388 | loss val 0.0869 | acc val 97.43 %
  epoque  7/9 | loss train 0.0315 | loss val 0.0867 | acc val 97.35 %
  epoque  8/9 | loss train 0.0249 | loss val 0.0806 | acc val 97.65 %
  epoque  9/9 | loss train 0.0207 | loss val 0.0769 | acc val 97.65 %

== mlp huge | Momentum 0.9 lr 0.01 | seed 4
  loss initiale : 2.5715  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2819 | loss val 0.1735 | acc val 95.17 %
  epoque  2/9 | loss train 0.1294 | loss val 0.1253 | acc val 96.47 %
  epoque  3/9 | loss train 0.0899 | loss val 0.1111 | acc val 96.93 %
  epoque  4/9 | loss train 0.0658 | loss val 0.1067 | acc val 96.85 %
  epoque  5/9 | loss train 0.0512 | loss val 0.0914 | acc val 97.33 %
  epoque  6/9 | loss train 0.0406 | loss val 0.0878 | acc val 97.32 %
  epoque  7/9 | loss train 0.0322 | loss val 0.0844 | acc val 97.37 %
  epoque  8/9 | loss train 0.0261 | loss val 0.0877 | acc val 97.35 %
  epoque  9/9 | loss train 0.0215 | loss val 0.0828 | acc val 97.53 %

== mlp huge | AdamW lr 1e-3 | seed 0
  loss initiale : 2.8207  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2151 | loss val 0.1189 | acc val 96.63 %
  epoque  2/9 | loss train 0.0792 | loss val 0.1017 | acc val 96.95 %
  epoque  3/9 | loss train 0.0498 | loss val 0.0919 | acc val 97.27 %
  epoque  4/9 | loss train 0.0337 | loss val 0.0807 | acc val 97.72 %
  epoque  5/9 | loss train 0.0245 | loss val 0.1022 | acc val 97.15 %
  epoque  6/9 | loss train 0.0212 | loss val 0.0956 | acc val 97.57 %
  epoque  7/9 | loss train 0.0186 | loss val 0.0949 | acc val 97.53 %
  epoque  8/9 | loss train 0.0152 | loss val 0.0874 | acc val 97.95 %
  epoque  9/9 | loss train 0.0175 | loss val 0.1172 | acc val 97.27 %

== mlp huge | AdamW lr 1e-3 | seed 1
  loss initiale : 2.6559  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2140 | loss val 0.1164 | acc val 96.53 %
  epoque  2/9 | loss train 0.0818 | loss val 0.1094 | acc val 96.63 %
  epoque  3/9 | loss train 0.0507 | loss val 0.0918 | acc val 97.42 %
  epoque  4/9 | loss train 0.0341 | loss val 0.0906 | acc val 97.53 %
  epoque  5/9 | loss train 0.0249 | loss val 0.0847 | acc val 97.57 %
  epoque  6/9 | loss train 0.0199 | loss val 0.0888 | acc val 97.57 %
  epoque  7/9 | loss train 0.0197 | loss val 0.0865 | acc val 97.58 %
  epoque  8/9 | loss train 0.0160 | loss val 0.0864 | acc val 98.00 %
  epoque  9/9 | loss train 0.0153 | loss val 0.1119 | acc val 97.62 %

== mlp huge | AdamW lr 1e-3 | seed 2
  loss initiale : 2.8004  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2132 | loss val 0.1162 | acc val 96.55 %
  epoque  2/9 | loss train 0.0806 | loss val 0.0931 | acc val 97.30 %
  epoque  3/9 | loss train 0.0482 | loss val 0.1004 | acc val 96.95 %
  epoque  4/9 | loss train 0.0340 | loss val 0.0817 | acc val 97.57 %
  epoque  5/9 | loss train 0.0242 | loss val 0.1055 | acc val 97.43 %
  epoque  6/9 | loss train 0.0195 | loss val 0.0969 | acc val 97.63 %
  epoque  7/9 | loss train 0.0171 | loss val 0.1031 | acc val 97.42 %
  epoque  8/9 | loss train 0.0168 | loss val 0.1122 | acc val 97.25 %
  epoque  9/9 | loss train 0.0178 | loss val 0.0970 | acc val 97.95 %

== mlp huge | AdamW lr 1e-3 | seed 3
  loss initiale : 2.5153  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2156 | loss val 0.1183 | acc val 96.45 %
  epoque  2/9 | loss train 0.0785 | loss val 0.0973 | acc val 97.07 %
  epoque  3/9 | loss train 0.0484 | loss val 0.0908 | acc val 97.28 %
  epoque  4/9 | loss train 0.0324 | loss val 0.0883 | acc val 97.75 %
  epoque  5/9 | loss train 0.0263 | loss val 0.1011 | acc val 97.28 %
  epoque  6/9 | loss train 0.0189 | loss val 0.0911 | acc val 97.75 %
  epoque  7/9 | loss train 0.0212 | loss val 0.1030 | acc val 97.32 %
  epoque  8/9 | loss train 0.0170 | loss val 0.1080 | acc val 97.40 %
  epoque  9/9 | loss train 0.0101 | loss val 0.1012 | acc val 97.58 %

== mlp huge | AdamW lr 1e-3 | seed 4
  loss initiale : 2.5715  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.2136 | loss val 0.1155 | acc val 96.50 %
  epoque  2/9 | loss train 0.0813 | loss val 0.0899 | acc val 97.32 %
  epoque  3/9 | loss train 0.0512 | loss val 0.0802 | acc val 97.52 %
  epoque  4/9 | loss train 0.0325 | loss val 0.0861 | acc val 97.55 %
  epoque  5/9 | loss train 0.0255 | loss val 0.0968 | acc val 97.48 %
  epoque  6/9 | loss train 0.0220 | loss val 0.0921 | acc val 97.50 %
  epoque  7/9 | loss train 0.0197 | loss val 0.0837 | acc val 97.90 %
  epoque  8/9 | loss train 0.0139 | loss val 0.1061 | acc val 97.50 %
  epoque  9/9 | loss train 0.0193 | loss val 0.1161 | acc val 97.50 %

== mlp huge | AdEMAMix lr 2e-4 | seed 0
  loss initiale : 2.8207  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3209 | loss val 0.1778 | acc val 94.85 %
  epoque  2/9 | loss train 0.1139 | loss val 0.1098 | acc val 96.77 %
  epoque  3/9 | loss train 0.0590 | loss val 0.0845 | acc val 97.63 %
  epoque  4/9 | loss train 0.0300 | loss val 0.0769 | acc val 97.65 %
  epoque  5/9 | loss train 0.0144 | loss val 0.0772 | acc val 97.85 %
  epoque  6/9 | loss train 0.0061 | loss val 0.0788 | acc val 98.10 %
  epoque  7/9 | loss train 0.0028 | loss val 0.0911 | acc val 98.02 %
  epoque  8/9 | loss train 0.0016 | loss val 0.0974 | acc val 98.10 %
  epoque  9/9 | loss train 0.0013 | loss val 0.1185 | acc val 97.80 %

== mlp huge | AdEMAMix lr 2e-4 | seed 1
  loss initiale : 2.6559  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3156 | loss val 0.1662 | acc val 95.20 %
  epoque  2/9 | loss train 0.1148 | loss val 0.0984 | acc val 97.05 %
  epoque  3/9 | loss train 0.0591 | loss val 0.0809 | acc val 97.60 %
  epoque  4/9 | loss train 0.0298 | loss val 0.0761 | acc val 97.80 %
  epoque  5/9 | loss train 0.0147 | loss val 0.0730 | acc val 97.87 %
  epoque  6/9 | loss train 0.0063 | loss val 0.0792 | acc val 97.90 %
  epoque  7/9 | loss train 0.0031 | loss val 0.0890 | acc val 97.87 %
  epoque  8/9 | loss train 0.0015 | loss val 0.1022 | acc val 97.88 %
  epoque  9/9 | loss train 0.0009 | loss val 0.1081 | acc val 98.13 %

== mlp huge | AdEMAMix lr 2e-4 | seed 2
  loss initiale : 2.8004  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3170 | loss val 0.1670 | acc val 95.17 %
  epoque  2/9 | loss train 0.1133 | loss val 0.1014 | acc val 97.22 %
  epoque  3/9 | loss train 0.0583 | loss val 0.0805 | acc val 97.55 %
  epoque  4/9 | loss train 0.0292 | loss val 0.0728 | acc val 97.90 %
  epoque  5/9 | loss train 0.0133 | loss val 0.0842 | acc val 97.80 %
  epoque  6/9 | loss train 0.0059 | loss val 0.0806 | acc val 97.87 %
  epoque  7/9 | loss train 0.0026 | loss val 0.0925 | acc val 97.87 %
  epoque  8/9 | loss train 0.0021 | loss val 0.0951 | acc val 98.08 %
  epoque  9/9 | loss train 0.0037 | loss val 0.1208 | acc val 97.73 %

== mlp huge | AdEMAMix lr 2e-4 | seed 3
  loss initiale : 2.5153  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3118 | loss val 0.1653 | acc val 95.30 %
  epoque  2/9 | loss train 0.1129 | loss val 0.1034 | acc val 96.88 %
  epoque  3/9 | loss train 0.0588 | loss val 0.0820 | acc val 97.48 %
  epoque  4/9 | loss train 0.0293 | loss val 0.0700 | acc val 97.93 %
  epoque  5/9 | loss train 0.0133 | loss val 0.0736 | acc val 98.10 %
  epoque  6/9 | loss train 0.0057 | loss val 0.0807 | acc val 97.97 %
  epoque  7/9 | loss train 0.0026 | loss val 0.0868 | acc val 98.13 %
  epoque  8/9 | loss train 0.0010 | loss val 0.0945 | acc val 98.12 %
  epoque  9/9 | loss train 0.0006 | loss val 0.1086 | acc val 98.12 %

== mlp huge | AdEMAMix lr 2e-4 | seed 4
  loss initiale : 2.5715  (log(10) = 2.3026)
  epoque  1/9 | loss train 0.3147 | loss val 0.1655 | acc val 95.18 %
  epoque  2/9 | loss train 0.1129 | loss val 0.1013 | acc val 96.98 %
  epoque  3/9 | loss train 0.0591 | loss val 0.0863 | acc val 97.42 %
  epoque  4/9 | loss train 0.0296 | loss val 0.0854 | acc val 97.65 %
  epoque  5/9 | loss train 0.0137 | loss val 0.0726 | acc val 97.93 %
  epoque  6/9 | loss train 0.0060 | loss val 0.0787 | acc val 98.18 %
  epoque  7/9 | loss train 0.0027 | loss val 0.0888 | acc val 97.97 %
  epoque  8/9 | loss train 0.0015 | loss val 0.0968 | acc val 97.97 %
  epoque  9/9 | loss train 0.0018 | loss val 0.1176 | acc val 97.75 %



