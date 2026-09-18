# Lab 3 

## Objectif

Utiliser des pipelines Hugging Face pré-entraînés pour explorer plusieurs tâches de NLP : classification d'opinion (sentiment), classification zero-shot, inférence question-réponse (QNLI), détection de paraphrases (QQP) et vérification grammaticale (CoLA).

## Structure du projet

```
Lab3-HuggingFace/
├── lab3_huggingface_pipelines.py
├── requirements.txt
└── lab3_analyse_recapitulative.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

requirements.txt :
```
transformers
torch
scikit-learn
```

## Exécution

```bash
python lab3_huggingface_pipelines.py
```

La première exécution télécharge plusieurs modèles pré-entraînés depuis Hugging Face Hub (plusieurs centaines de Mo au total). Une connexion internet stable est nécessaire, et l'exécution peut prendre plusieurs minutes.

---

## Résultats obtenus par étape

### Étape 2 — Classification d'opinion (sentiment analysis)

Modèle : distilbert-base-uncased-finetuned-sst-2-english

Inférence unitaire : "I really liked the movie!!" → POSITIVE (score = 0.9998)

Traitement en lot :

| Phrase | Prédiction | Score |
|---|---|---|
| I really liked the movie!! | POSITIVE | 0.9998 |
| Great job ruining my day. | NEGATIVE | 0.8667 |
| This product exceeded my expectations. | POSITIVE | 0.9989 |
| Wow, just what I needed... another problem. | POSITIVE | 0.9863 |
| Absolutely fantastic experience! | POSITIVE | 0.9999 |

Accuracy = 0.80

La phrase sarcastique "Wow, just what I needed... another problem." est mal classée (prédite POSITIVE au lieu de NEGATIVE) : le modèle est sensible au ton positif des mots ("Wow", "just what I needed") sans détecter le sarcasme.

### Étape 3 — Classification zero-shot

Modèle : MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

Texte : "The national football team won the cup yesterday."

| Label | Score |
|---|---|
| sports | 0.9918 |
| technology | 0.0062 |
| health | 0.0020 |

Avec labels reformulés (sports, tech news, health) :

| Label | Score |
|---|---|
| sports | 0.9975 |
| health | 0.0020 |
| tech news | 0.0005 |

Test multilingue (français) : "L'équipe nationale de football a remporté la coupe hier." → sports = 0.9974 (résultat quasi identique à l'anglais)

L'ordre des labels ne change pas les scores, mais leur formulation influence les résultats (technology → 0.0062 vs tech news → 0.0005). Le modèle est robuste en multilingue sur cet exemple simple.

### Étape 4 — Inférence question-réponse (QNLI)

Modèle : cross-encoder/qnli-electra-base

| Question | Attendu | Prédit |
|---|---|---|
| Where do penguins live? | LABEL_1 | LABEL_0 |
| What is the capital of Paris? | LABEL_0 | LABEL_0 |
| When was the Eiffel Tower built? | LABEL_1 | LABEL_0 |
| Who wrote Romeo and Juliet? | LABEL_0 | LABEL_0 |
| What is Python used for? | LABEL_1 | LABEL_0 |

Accuracy QNLI = 0.40

Le modèle prédit systématiquement LABEL_0 ("pas de réponse"), même quand le passage contient clairement l'information. Cela illustre que la performance d'un modèle pré-entraîné dépend fortement du couplage entre le format des passages/questions du jeu de test et celui utilisé lors de l'entraînement du modèle.

### Étape 5 — Détection de paraphrases (QQP)

Modèle : textattack/bert-base-uncased-QQP

| Question 1 | Question 2 | Attendu | Prédit |
|---|---|---|---|
| How can I learn Python? | What is the best way to study Python? | LABEL_1 | LABEL_1 |
| What is the capital of France? | How can I learn Python? | LABEL_0 | LABEL_0 |
| How do I reset my password? | What is the process to change my password? | LABEL_1 | LABEL_1 |
| What is the weather today? | How do I reset my password? | LABEL_0 | LABEL_0 |
| What is the best smartphone in 2024? | Which smartphone should I buy in 2024? | LABEL_1 | LABEL_1 |
| How old is the Eiffel Tower? | What is the best smartphone in 2024? | LABEL_0 | LABEL_0 |
| How can I improve my English speaking? | What are ways to get better at speaking English? | LABEL_1 | LABEL_1 |
| What is the tallest mountain in the world? | How can I improve my English speaking? | LABEL_0 | LABEL_0 |

Accuracy QQP = 1.00. Le modèle distingue parfaitement les paraphrases des paires non liées sur ce mini-corpus.

### Étape 6 — Vérification grammaticale (CoLA)

Modèle : textattack/distilbert-base-uncased-CoLA

| Phrase | Prédiction | Score |
|---|---|---|
| The cat sat on the mat. | acceptée (correcte) | 0.992 |
| The cat on sat mat the. | rejetée (incorrecte) | 0.963 |
| She is going to the market tomorrow. | acceptée (correcte) | 0.977 |
| Going market she tomorrow to is the. | rejetée (incorrecte) | 0.973 |
| He don't like apples. | acceptée (correcte) | 0.621 |
| The colorless green ideas sleep furiously. | acceptée (correcte) | 0.527 |
| They was late for the meeting. | acceptée (correcte) | 0.977 |
| The sun rises in the east. | acceptée (correcte) | 0.989 |

"The colorless green ideas sleep furiously" est acceptée : la phrase est grammaticalement correcte bien que sémantiquement absurde, ce qui montre que CoLA juge la structure, pas le sens.

"He don't like apples." est acceptée avec un score modéré (0.621), alors qu'elle contient une faute d'accord sujet-verbe.

"They was late for the meeting." est acceptée à tort (0.977) alors qu'elle contient une erreur d'accord ("was" au lieu de "were").

---

## Voir aussi

Le fichier lab3_analyse_recapitulative.md contient l'analyse comparative complète (Étape 7) entre les cinq tâches, leurs limites respectives et des pistes d'amélioration.

<img width="1280" height="676" alt="A1" src="https://github.com/user-attachments/assets/55071de9-1b24-4a64-86c8-37ef3c8f298d" />

<img width="1280" height="670" alt="A2" src="https://github.com/user-attachments/assets/8ad99e2f-82ba-4fbc-a363-75430fbc2823" />

<img width="1277" height="668" alt="A3" src="https://github.com/user-attachments/assets/9c63c827-5bed-46e7-87b1-16021f256106" />

<img width="1280" height="672" alt="A4" src="https://github.com/user-attachments/assets/53bf237e-7780-4712-a3c9-8540aa17fa40" />

<img width="1280" height="672" alt="A5" src="https://github.com/user-attachments/assets/772004cd-a7aa-4795-8d92-cea36f77e95a" />

<img width="1280" height="667" alt="A6" src="https://github.com/user-attachments/assets/599a72f2-df8c-4b53-b0e2-6c2f8039888a" />

<img width="1280" height="672" alt="A7" src="https://github.com/user-attachments/assets/bd6205d0-2367-4e48-9474-dc410c7396b7" />

<img width="1280" height="667" alt="A8" src="https://github.com/user-attachments/assets/8d44dfbb-932e-49ca-970f-dc1f5b5d7bd4" />

<img width="1280" height="663" alt="A9" src="https://github.com/user-attachments/assets/fecce5ea-643a-4231-8ef3-b8da4780850b" />
