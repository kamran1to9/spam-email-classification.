import os
import json
import time

import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 64
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-5

# Limit training work to make CPU training faster.
MAX_TRAIN_SAMPLES = 1200

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs("results", exist_ok=True)

print("=" * 50)
print("DISTILBERT SPAM CLASSIFICATION")
print("=" * 50)

print("Device:", DEVICE)
print("Maximum sequence length:", MAX_LENGTH)
print("Epochs:", EPOCHS)
print("Training sample limit:", MAX_TRAIN_SAMPLES)


# ==========================================
# 2. LOAD AND CLEAN DATASET
# ==========================================

df = pd.read_csv("email.csv")

df["Category"] = (
    df["Category"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["Message"] = (
    df["Message"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    df["Category"].isin(["ham", "spam"])
].copy()

df = df[df["Message"] != ""].copy()

df["label"] = df["Category"].map({
    "ham": 0,
    "spam": 1
})

df["label"] = df["label"].astype(int)

print("\nDataset size:", len(df))

print("\nClass distribution:")
print(df["Category"].value_counts())


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    df["Message"].tolist(),
    df["label"].tolist(),
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("\nOriginal training samples:", len(X_train))
print("Test samples:", len(X_test))


# ==========================================
# 4. REDUCE TRAINING SIZE FOR CPU
# ==========================================

if len(X_train) > MAX_TRAIN_SAMPLES:

    train_df = pd.DataFrame({
        "Message": X_train,
        "label": y_train
    })

    train_df, _ = train_test_split(
        train_df,
        train_size=MAX_TRAIN_SAMPLES,
        random_state=42,
        stratify=train_df["label"]
    )

    X_train = train_df["Message"].tolist()
    y_train = train_df["label"].tolist()

print("\nTraining samples used:", len(X_train))
print("Test samples used:", len(X_test))


# ==========================================
# 5. LOAD TOKENIZER
# ==========================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ==========================================
# 6. TOKENIZE ONCE
# ==========================================

print("Tokenizing training messages...")

train_encodings = tokenizer(
    X_train,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

print("Tokenizing test messages...")

test_encodings = tokenizer(
    X_test,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt"
)


# ==========================================
# 7. PYTORCH DATASET
# ==========================================

class EmailDataset(Dataset):

    def __init__(self, encodings, labels):

        self.encodings = encodings

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return {
            "input_ids":
                self.encodings["input_ids"][index],

            "attention_mask":
                self.encodings["attention_mask"][index],

            "labels":
                self.labels[index]
        }


train_dataset = EmailDataset(
    train_encodings,
    y_train
)

test_dataset = EmailDataset(
    test_encodings,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# 8. LOAD PRETRAINED DISTILBERT
# ==========================================

print("\nLoading pretrained DistilBERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 9. CUSTOM PYTORCH TRAINING LOOP
# ==========================================

training_losses = []

start_time = time.time()

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch_number, batch in enumerate(train_loader, start=1):

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if batch_number % 25 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_number}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_loader)

    training_losses.append(average_loss)

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS} completed. "
        f"Average loss: {average_loss:.4f}\n"
    )

training_time = time.time() - start_time


# ==========================================
# 10. EVALUATION
# ==========================================

print("Evaluating DistilBERT...")

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )


# ==========================================
# 11. CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

print("\n" + "=" * 50)
print("DISTILBERT EVALUATION RESULTS")
print("=" * 50)

print(f"Accuracy:       {accuracy:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"F1-score:       {f1:.4f}")
print(f"Training time:  {training_time:.2f} seconds")

print("\nClassification Report:")

report = classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1],
    target_names=["HAM", "SPAM"],
    zero_division=0
)

print(report)

with open(
    "results/distilbert_classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)


# ==========================================
# 12. SAVE METRICS
# ==========================================

metrics = {
    "model": "DistilBERT",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "training_time_seconds": training_time,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "learning_rate": LEARNING_RATE,
    "training_samples_used": len(X_train),
    "test_samples": len(X_test),
    "device": str(DEVICE)
}

with open(
    "results/distilbert_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)

pd.DataFrame({
    "actual_label": all_labels,
    "predicted_label": all_predictions
}).to_csv(
    "results/distilbert_predictions.csv",
    index=False
)


# ==========================================
# 13. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("DistilBERT Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks([0, 1], ["HAM", "SPAM"])
plt.yticks([0, 1], ["HAM", "SPAM"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "results/distilbert_confusion_matrix.png",
    dpi=200
)

plt.close()


# ==========================================
# 14. TRAINING LOSS GRAPH
# ==========================================

plt.figure(figsize=(7, 5))

plt.plot(
    range(1, EPOCHS + 1),
    training_losses,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")

plt.title("DistilBERT Training Loss")

plt.xticks(range(1, EPOCHS + 1))

plt.tight_layout()

plt.savefig(
    "results/distilbert_training_loss.png",
    dpi=200
)

plt.close()


# ==========================================
# 15. SAVE MODEL AND TOKENIZER
# ==========================================

print("\nSaving trained model...")

model.save_pretrained(
    "distilbert_saved_model"
)

tokenizer.save_pretrained(
    "distilbert_saved_model"
)


# ==========================================
# 16. COMPLETION
# ==========================================

print("\n" + "=" * 50)
print("DISTILBERT EXPERIMENT COMPLETED")
print("=" * 50)

print("Metrics: results/distilbert_metrics.json")
print("Report: results/distilbert_classification_report.txt")
print("Predictions: results/distilbert_predictions.csv")
print("Confusion matrix: results/distilbert_confusion_matrix.png")
print("Training loss: results/distilbert_training_loss.png")
print("Saved model: distilbert_saved_model/")