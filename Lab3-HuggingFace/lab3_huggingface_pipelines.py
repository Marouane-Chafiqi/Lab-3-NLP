# -*- coding: utf-8 -*-
"""
Lab 3 — Pipelines Hugging Face pour la classification de textes :
du sentiment au zero-shot
====================================================================
Ce script réalise l'ensemble des tâches (Étapes 2 à 6) du TP :
  - Étape 2 : Classification d'opinion (sentiment analysis)
  - Étape 3 : Classification zero-shot
  - Étape 4 : Inférence question-réponse (QNLI)
  - Étape 5 : Détection de paraphrases (QQP)
  - Étape 6 : Vérification grammaticale (CoLA)
  - Étape 7 : voir le fichier séparé lab3_analyse_recapitulative.md

IMPORTANT :
  - Ce script nécessite une connexion internet au premier lancement pour
    télécharger les modèles depuis huggingface.co (les modèles sont ensuite
    mis en cache localement, généralement dans ~/.cache/huggingface/).
  - L'inférence se fait sur CPU par défaut si aucun GPU n'est disponible :
    c'est normal et attendu, mais un peu plus lent.
  - Le premier téléchargement de chaque modèle peut prendre plusieurs minutes
    selon la connexion (les modèles font entre 250 Mo et 700 Mo chacun).
"""

from transformers import pipeline
from sklearn.metrics import accuracy_score


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ==============================================================================
# ETAPE 2 — Classification d'opinion (sentiment analysis)
# ==============================================================================
section("ETAPE 2 — Classification d'opinion (sentiment analysis)")

classification_pipeline = pipeline(
    task="sentiment-analysis",  # alias de text-classification
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

# --- Inférence unitaire ---
result = classification_pipeline("I really liked the movie!!")
print("\nInférence unitaire :")
print(result)

# --- Traitement en lot ---
texts = [
    "I really liked the movie!!",
    "Great job ruining my day.",
    "This product exceeded my expectations.",
    "Wow, just what I needed... another problem.",
    "Absolutely fantastic experience!",
]
results = classification_pipeline(texts)
print("\nTraitement en lot :")
for t, r in zip(texts, results):
    print(f"  {t!r:55s} -> {r['label']:8s} (score={r['score']:.4f})")

# --- Évaluation (accuracy) ---
true_labels = ["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE"]
predicted = [res["label"] for res in results]
acc = accuracy_score(true_labels, predicted)
print(f"\nAccuracy = {acc:.2f}  (critère de réussite : >= 0.80)")


# ==============================================================================
# ETAPE 3 — Classification zero-shot
# ==============================================================================
section("ETAPE 3 — Classification zero-shot")

zero_shot_classifier = pipeline(
    task="zero-shot-classification",
    model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
)

text = "The national football team won the cup yesterday."
candidate_labels = ["sports", "technology", "health"]
result = zero_shot_classifier(text, candidate_labels)
print("\nTexte :", text)
print("Résultat :")
for label, score in zip(result["labels"], result["scores"]):
    print(f"  {label:12s} {score:.4f}")

# --- Variation des labels : reformulation ---
candidate_labels_v2 = ["sports", "tech news", "health"]
result_v2 = zero_shot_classifier(text, candidate_labels_v2)
print("\nAvec labels reformulés", candidate_labels_v2, ":")
for label, score in zip(result_v2["labels"], result_v2["scores"]):
    print(f"  {label:12s} {score:.4f}")

# --- Test en français (robustesse multilingue du modèle) ---
text_fr = "L'équipe nationale de football a remporté la coupe hier."
result_fr = zero_shot_classifier(text_fr, candidate_labels)
print("\nTexte français :", text_fr)
print("Résultat :")
for label, score in zip(result_fr["labels"], result_fr["scores"]):
    print(f"  {label:12s} {score:.4f}")

print("""
Analyse : l'ordre des labels ne change pas les scores, mais leur formulation
(ex. "technology" vs "tech news") peut influencer fortement les résultats.
""")


# ==============================================================================
# ETAPE 4 — Inférence question-réponse (QNLI)
# ==============================================================================
section("ETAPE 4 — Inférence question-réponse (QNLI)")

qnli_pipeline = pipeline(
    task="text-classification",
    model="cross-encoder/qnli-electra-base",
)

passage = "Penguins are found primarily in the Southern Hemisphere."
question1 = "Where do penguins live?"
question2 = "What is the capital of Paris?"

result1 = qnli_pipeline({"text": question1, "text_pair": passage})
result2 = qnli_pipeline({"text": question2, "text_pair": passage})

label_map_qnli = {"LABEL_0": "Pas de réponse", "LABEL_1": "Réponse présente"}
print(f"\nQ1: {question1!r}")
print(f"  -> {result1}  ({label_map_qnli.get(result1['label'], result1['label'])})")
print(f"Q2: {question2!r}")
print(f"  -> {result2}  ({label_map_qnli.get(result2['label'], result2['label'])})")

# --- Mini-jeu d'essai (5 passages / 5 questions) ---
mini_dataset_qnli = [
    {
        "question": "Where do penguins live?",
        "passage": "Penguins are found primarily in the Southern Hemisphere.",
        "true_label": "LABEL_1",
    },
    {
        "question": "What is the capital of Paris?",
        "passage": "Penguins are found primarily in the Southern Hemisphere.",
        "true_label": "LABEL_0",
    },
    {
        "question": "When was the Eiffel Tower built?",
        "passage": "The Eiffel Tower was completed in 1889 in Paris, France.",
        "true_label": "LABEL_1",
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "passage": "The Eiffel Tower was completed in 1889 in Paris, France.",
        "true_label": "LABEL_0",
    },
    {
        "question": "What is Python used for?",
        "passage": "Python is a popular programming language widely used for data science and web development.",
        "true_label": "LABEL_1",
    },
]

qnli_true = [ex["true_label"] for ex in mini_dataset_qnli]
qnli_pred = [
    qnli_pipeline({"text": ex["question"], "text_pair": ex["passage"]})["label"]
    for ex in mini_dataset_qnli
]
print("\nMini-jeu d'essai QNLI :")
for ex, pred in zip(mini_dataset_qnli, qnli_pred):
    print(f"  Q: {ex['question']!r:45s} attendu={ex['true_label']} prédit={pred}")
print(f"\nAccuracy QNLI = {accuracy_score(qnli_true, qnli_pred):.2f}")


# ==============================================================================
# ETAPE 5 — Détection de paraphrases (QQP)
# ==============================================================================
section("ETAPE 5 — Détection de paraphrases (QQP)")

qqp_pipeline = pipeline(
    task="text-classification",
    model="textattack/bert-base-uncased-QQP",
)

label_map_qqp = {"LABEL_0": "not_paraphrase", "LABEL_1": "paraphrase"}

question1 = "How can I learn Python?"
question2 = "What is the best way to study Python?"
result = qqp_pipeline({"text": question1, "text_pair": question2})
print(f"\nPaire similaire : {question1!r} / {question2!r}")
print(f"  -> {result}  ({label_map_qqp.get(result['label'], result['label'])})")

q3 = "What is the capital of France?"
result2 = qqp_pipeline({"text": question1, "text_pair": q3})
print(f"\nPaire dissemblable : {question1!r} / {q3!r}")
print(f"  -> {result2}  ({label_map_qqp.get(result2['label'], result2['label'])})")

# --- Corpus de 8 couples de questions ---
qqp_corpus = [
    ("How can I learn Python?", "What is the best way to study Python?", "LABEL_1"),
    ("What is the capital of France?", "How can I learn Python?", "LABEL_0"),
    ("How do I reset my password?", "What is the process to change my password?", "LABEL_1"),
    ("What is the weather today?", "How do I reset my password?", "LABEL_0"),
    ("What is the best smartphone in 2024?", "Which smartphone should I buy in 2024?", "LABEL_1"),
    ("How old is the Eiffel Tower?", "What is the best smartphone in 2024?", "LABEL_0"),
    ("How can I improve my English speaking?", "What are ways to get better at speaking English?", "LABEL_1"),
    ("What is the tallest mountain in the world?", "How can I improve my English speaking?", "LABEL_0"),
]

qqp_true = [c[2] for c in qqp_corpus]
qqp_pred = [
    qqp_pipeline({"text": c[0], "text_pair": c[1]})["label"] for c in qqp_corpus
]
print("\nCorpus de paires de questions :")
for (q1, q2, true), pred in zip(qqp_corpus, qqp_pred):
    ok = "OK" if pred == true else "ERREUR"
    print(f"  [{ok:6s}] {q1!r:45s} <-> {q2!r:45s} attendu={true} prédit={pred}")
print(f"\nAccuracy QQP = {accuracy_score(qqp_true, qqp_pred):.2f}")


# ==============================================================================
# ETAPE 6 — Vérification grammaticale (CoLA)
# ==============================================================================
section("ETAPE 6 — Vérification grammaticale (CoLA)")

cola_classifier = pipeline(
    task="text-classification",
    model="textattack/distilbert-base-uncased-CoLA",
)

label_map_cola = {"LABEL_0": "rejetée (incorrecte)", "LABEL_1": "acceptée (correcte)"}

sentence_ok = "The cat sat on the mat."
sentence_ko = "The cat on sat mat the."
print(f"\n{sentence_ok!r}")
r_ok = cola_classifier(sentence_ok)
print(f"  -> {r_ok}  ({label_map_cola.get(r_ok[0]['label'], r_ok[0]['label'])})")

print(f"\n{sentence_ko!r}")
r_ko = cola_classifier(sentence_ko)
print(f"  -> {r_ko}  ({label_map_cola.get(r_ko[0]['label'], r_ko[0]['label'])})")

# --- Mini corpus de phrases (correctes / incorrectes / correctes mais absurdes) ---
cola_corpus = [
    "The cat sat on the mat.",
    "The cat on sat mat the.",
    "She is going to the market tomorrow.",
    "Going market she tomorrow to is the.",
    "He don't like apples.",  # faute d'accord sujet-verbe
    "The colorless green ideas sleep furiously.",  # grammaticalement correcte mais absurde
    "They was late for the meeting.",  # faute d'accord
    "The sun rises in the east.",
]

print("\nMini corpus CoLA :")
for s in cola_corpus:
    r = cola_classifier(s)[0]
    label = label_map_cola.get(r["label"], r["label"])
    print(f"  {s!r:55s} -> {r['label']:10s} ({label}, score={r['score']:.3f})")

print("""
Remarque : CoLA juge la grammaticalité, pas le sens. La phrase "The colorless
green ideas sleep furiously" est grammaticalement correcte bien que
sémantiquement absurde — un bon test pour vérifier que le modèle regarde
la structure et non le sens.
""")


print("\n" + "=" * 70)
print("Script terminé. Voir aussi : lab3_analyse_recapitulative.md (Étape 7)")
print("=" * 70)
