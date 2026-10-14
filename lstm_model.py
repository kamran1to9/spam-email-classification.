# ============================================================
# LSTM EMAIL SPAM CLASSIFICATION
# ============================================================

import json
import pickle
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "email.csv"
MODEL_PATH = BASE_DIR / "lstm_spam_model.pth"
VECTORIZER_PATH = BASE_DIR / "lstm_vectorizer.pkl"
CONFIG_PATH = BASE_DIR / "lstm_model_config.json"

RANDOM_SEED = 42
MAX_FEATURES = 5000
MAX_LENGTH = 100
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

PAD_ID = 0
UNK_ID = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# 2. CLEAN TEXT
# ============================================================

def clean_text(text):
    """Normalize an email message before tokenization."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# 3. LOAD AND VALIDATE DATASET
# ============================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {DATASET_PATH.name}. Put email.csv in the same folder as this script."
    )

df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.astype(str).str.strip()

required_columns = {"Category", "Message"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(
        f"Missing required column(s): {sorted(missing_columns)}. "
        f"Found columns: {df.columns.tolist()}"
    )

# Drop missing messages BEFORE converting values to strings.
df = df.dropna(subset=["Message"]).copy()
df["Category"] = df["Category"].astype(str).str.strip().str.lower()
df = df[df["Category"].isin(["ham", "spam"])].copy()
df["Message"] = df["Message"].astype(str).apply(clean_text)
df = df[df["Message"].str.len() > 0].copy()
df["label"] = df["Category"].map({"ham": 0, "spam": 1}).astype(int)

if df.empty or df["label"].nunique() != 2:
    raise ValueError("The dataset must contain non-empty messages from both ham and spam classes.")

print("\nDATASET INFORMATION")
print("-" * 45)
print(f"Dataset shape: {df.shape}")
print(df["Category"].value_counts().to_string())

messages = df["Message"].tolist()
labels = df["label"].to_numpy(dtype=np.int64)


# ============================================================
# 4. TRAIN/TEST SPLIT
# ============================================================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    messages,
    labels,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=labels,
)

print("\nDATA SPLIT")
print("-" * 45)
print(f"Training messages: {len(X_train_text)}")
print(f"Testing messages:  {len(X_test_text)}")


# ============================================================
# 5. BUILD VOCABULARY FROM TRAINING DATA ONLY
# ============================================================

vectorizer = CountVectorizer(
    max_features=MAX_FEATURES,
    token_pattern=r"(?u)\b\w+\b",
)
vectorizer.fit(X_train_text)
vocab = vectorizer.vocabulary_
analyzer = vectorizer.build_analyzer()
vocab_size = len(vocab) + 2

print(f"\nVocabulary size (including PAD/UNK): {vocab_size}")


# ============================================================
# 6. CONVERT TEXT TO PADDED SEQUENCES AND LENGTHS
# ============================================================

def text_to_sequence(text):
    tokens = analyzer(text)
    # Known tokens use IDs 2..vocab_size-1; unknown tokens use UNK_ID=1.
    sequence = [vocab[token] + 2 if token in vocab else UNK_ID for token in tokens]
    return sequence[:MAX_LENGTH]


def convert_messages(text_list):
    sequences = [text_to_sequence(text) for text in text_list]
    padded = torch.full((len(sequences), MAX_LENGTH), PAD_ID, dtype=torch.long)
    lengths = torch.zeros(len(sequences), dtype=torch.long)

    for i, sequence in enumerate(sequences):
        if sequence:
            padded[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            lengths[i] = len(sequence)

    return padded, lengths


X_train, train_lengths = convert_messages(X_train_text)
X_test, test_lengths = convert_messages(X_test_text)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

print("\nSEQUENCE INFORMATION")
print("-" * 45)
print(f"Training tensor shape: {tuple(X_train.shape)}")
print(f"Testing tensor shape:  {tuple(X_test.shape)}")
print(f"First 5 training lengths: {train_lengths[:5].tolist()}")


# ============================================================
# 7. DATA LOADER
# ============================================================

train_dataset = TensorDataset(X_train, train_lengths, y_train_tensor)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# ============================================================
# 8. LSTM MODEL
# ============================================================

class SpamLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        # pack_padded_sequence requires every length to be at least 1.
        safe_lengths = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(
            embedded,
            safe_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        output = self.dropout(hidden[-1])
        return self.fc(output).squeeze(1)


# ============================================================
# 9. INITIALIZE MODEL AND LOSS
# ============================================================

model = SpamLSTM(vocab_size)
ham_count = int(np.sum(y_train == 0))
spam_count = int(np.sum(y_train == 1))

print("\nTRAINING CLASS DISTRIBUTION")
print("-" * 45)
print(f"Ham messages:  {ham_count}")
print(f"Spam messages: {spam_count}")

if ham_count == 0 or spam_count == 0:
    raise ValueError("Training split must contain both ham and spam messages.")

# Mild class weighting helps with the smaller spam class without over-weighting it too much.
spam_weight = min(ham_count / spam_count, 5.0)
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([spam_weight], dtype=torch.float32)
)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5,
)


# ============================================================
# 10. TRAIN
# ============================================================

total_start = time.time()
print("\nMODEL TRAINING")
print("-" * 45)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for messages_batch, lengths_batch, labels_batch in train_loader:
        optimizer.zero_grad()
        logits = model(messages_batch, lengths_batch)
        loss = criterion(logits, labels_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / max(len(train_loader), 1)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {average_loss:.4f}")

training_time = time.time() - total_start


# ============================================================
# 11. EVALUATE ON HELD-OUT TEST DATA
# ============================================================

model.eval()
with torch.no_grad():
    logits = model(X_test, test_lengths)
    probabilities = torch.sigmoid(logits).cpu().numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

print("\nLSTM MODEL RESULTS")
print("=" * 45)
print(f"Accuracy:       {accuracy:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"F1-score:       {f1:.4f}")
print(f"Training time:  {training_time:.2f} seconds")

print("\nCONFUSION MATRIX (rows=true [Ham, Spam], columns=predicted [Ham, Spam])")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Ham", "Spam"],
    zero_division=0,
))

print("PREDICTION DISTRIBUTION")
unique_predictions, prediction_counts = np.unique(predictions, return_counts=True)
counts = {int(label): int(count) for label, count in zip(unique_predictions, prediction_counts)}
print(f"Ham:  {counts.get(0, 0)}")
print(f"Spam: {counts.get(1, 0)}")
print("\nFIRST 10 SPAM PROBABILITIES")
print(np.round(probabilities[:10].flatten(), 4))


# ============================================================
# 12. SAVE MODEL AND PREPROCESSING FILES
# ============================================================

# Save state_dict under the familiar filename.
torch.save(model.state_dict(), MODEL_PATH)

# Save the exact vectorizer used to create the vocabulary.
with open(VECTORIZER_PATH, "wb") as file:
    pickle.dump(vectorizer, file)

# Save architecture and preprocessing settings for inference scripts.
config = {
    "vocab_size": int(vocab_size),
    "embedding_dim": EMBEDDING_DIM,
    "hidden_dim": HIDDEN_DIM,
    "max_length": MAX_LENGTH,
    "pad_id": PAD_ID,
    "unk_id": UNK_ID,
    "threshold": 0.5,
    "labels": {"ham": 0, "spam": 1},
}
with open(CONFIG_PATH, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=4)

print("\nSAVED FILES")
print("-" * 45)
print(f"Model:      {MODEL_PATH.name}")
print(f"Vectorizer: {VECTORIZER_PATH.name}")
print(f"Config:     {CONFIG_PATH.name}")
print("Training and evaluation completed successfully.")
