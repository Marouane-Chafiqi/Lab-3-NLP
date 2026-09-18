# Étape 7 — Analyse récapitulative et discussion

## Comparaison des tâches

| Tâche | Type de sortie | Contextualisation | Sensibilité principale |
|---|---|---|---|
| **Sentiment analysis** | POSITIVE / NEGATIVE + score | Faible (phrase isolée, pas de contexte du discours) | Sarcasme, ponctuation, négation |
| **Zero-shot** | Labels + scores (softmax) | Moyenne (le modèle infère depuis le sens des labels) | Formulation des labels candidats |
| **QNLI** | LABEL_0 / LABEL_1 (binaire) | Bonne (compare question et passage) | Réponses implicites ou partielles |
| **QQP** | LABEL_0 / LABEL_1 (binaire) | Bonne (compare deux formulations) | Paraphrases partielles, mots-clés différents |
| **CoLA** | LABEL_0 / LABEL_1 (binaire) | Aucune (juge la forme, pas le sens) | Phrases correctes mais absurdes sémantiquement |

## Interprétation des limites

**Sentiment analysis (DistilBERT/SST-2).** Le modèle n'a pas de classe "neutre" — il force une décision binaire POSITIVE/NEGATIVE même sur des phrases ambiguës. Il est également sensible au sarcasme : une phrase comme *"Great job ruining my day"* contient des mots positifs ("great", "job") mais un sens négatif, ce qui peut tromper le modèle si celui-ci ne capture pas assez bien l'ironie.

**Zero-shot (DeBERTa-v3 mnli-fever-anli).** Le modèle transforme la classification en tâche d'inférence textuelle (le texte "implique-t-il" le label ?). Cela le rend flexible (pas besoin de ré-entraînement pour de nouvelles catégories) mais très sensible à la formulation exacte des labels : "technology" et "tech news" peuvent donner des scores différents pour le même texte, car le modèle compare des phrases entières générées à partir des labels (ex. "This text is about technology.").

**QNLI.** La sortie binaire (réponse présente / absente) ne donne aucune indication sur la localisation de la réponse dans le passage, ni sur son degré de certitude au-delà du score. Pour des passages longs contenant plusieurs informations, le modèle peut manquer une réponse partielle ou implicite.

**QQP.** La détection de paraphrases est également binaire, ce qui pose problème pour les cas ambigus (paraphrases partielles, où seule une partie du sens est partagée). Le modèle peut aussi être trompé par des questions qui partagent du vocabulaire mais un sens différent, ou l'inverse (formulations très différentes mais sens identique).

**CoLA.** Le modèle évalue uniquement la grammaticalité, pas le sens. Une phrase peut être jugée "acceptée" tout en étant sémantiquement absurde (ex. *"Colorless green ideas sleep furiously"*, exemple classique de Chomsky). À l'inverse, des phrases informelles mais compréhensibles (langage familier, tournures orales) peuvent être rejetées à tort car elles s'éloignent des normes grammaticales apprises par le modèle.

## Pistes d'amélioration

1. **Modèles multilingues** — pour les tâches nécessitant un support hors anglais (ex. `joeddav/xlm-roberta-large-xnli` pour le zero-shot multilingue), avec des labels formulés dans la même langue que le texte analysé.
2. **Fine-tuning sur des données spécifiques au domaine** — un modèle de sentiment entraîné sur des avis de films (SST-2) peut mal généraliser à des tweets, des avis produits, ou du texte médical ; un ré-entraînement (fine-tuning) sur des données du domaine cible améliore la robustesse.
3. **Calibration des seuils** — pour QQP et QNLI, au lieu de se fier uniquement au label binaire, utiliser le score de probabilité et ajuster un seuil de décision selon le contexte applicatif (ex. seuil plus strict si les faux positifs sont coûteux).
4. **Ensembles de modèles / vote majoritaire** — combiner plusieurs modèles (ex. plusieurs architectures de sentiment analysis) pour réduire l'impact des erreurs individuelles, notamment sur les cas ambigus (sarcasme, négation).
5. **Prise en compte du contexte** — pour le sentiment analysis, considérer la phrase dans son contexte (paragraphe entier, fil de conversation) plutôt qu'isolément, ce qui nécessite des modèles à plus grande fenêtre de contexte.
