import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================
# 1. LOAD AND CLEAN DATASET
# =====================================

df = pd.read_csv("email.csv")

# Remove invalid category labels
df["Category"] = (
    df["Category"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df = df[df["Category"].isin(["ham", "spam"])].copy()

# Remove missing and empty messages
df = df.dropna(subset=["Message"])
df["Message"] = df["Message"].astype(str)
df = df[df["Message"].str.strip() != ""].copy()

# Remove exact duplicate records
df = df.drop_duplicates(subset=["Category", "Message"]).copy()

# Convert labels into binary values
# ham = 0, spam = 1
df["Label"] = df["Category"].map({
    "ham": 0,
    "spam": 1
})

print("Dataset shape:", df.shape)

print("\nCategory distribution:")
print(df["Category"].value_counts())

# =====================================
# 2. SPLIT DATASET
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    df["Message"],
    df["Label"],
    test_size=0.20,
    random_state=42,
    stratify=df["Label"]
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# =====================================
# 3. TF-IDF VECTORIZATION
# =====================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=None,
    ngram_range=(1, 2),
    max_features=20000,
    sublinear_tf=True
)

# Fit only on training messages to prevent data leakage
X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)

# =====================================
# 4. DEFINE MODELS
# =====================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Multinomial Naive Bayes": MultinomialNB(
        alpha=1.0
    )
}

results = []
confusion_matrices = {}
training_times = {}

# =====================================
# 5. TRAIN AND EVALUATE MODELS
# =====================================

for model_name, model in models.items():

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    # Measure training time
    start_time = time.perf_counter()

    model.fit(X_train_tfidf, y_train)

    training_time = time.perf_counter() - start_time

    training_times[model_name] = training_time

    # Predictions
    y_pred = model.predict(X_test_tfidf)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Training Time (seconds)": training_time
    })

    confusion_matrices[model_name] = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Training time: {training_time:.4f} seconds")

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Ham", "Spam"],
        zero_division=0
    ))

# =====================================
# 6. SAVE RESULTS
# =====================================

results_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "traditional_ml_results.csv",
    index=False
)

# =====================================
# 7. METRIC COMPARISON VISUALIZATION
# =====================================

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]

plot_df = results_df.melt(
    id_vars="Model",
    value_vars=metric_columns,
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=plot_df,
    x="Metric",
    y="Score",
    hue="Model"
)

plt.title("Traditional Machine Learning Model Comparison")
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.xlabel("Evaluation Metric")
plt.tight_layout()

plt.savefig(
    "traditional_metrics_comparison.png",
    dpi=300
)

plt.show()

# =====================================
# 8. CONFUSION MATRICES
# =====================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

for ax, (model_name, matrix) in zip(
    axes,
    confusion_matrices.items()
):

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=ax
    )

    ax.set_title(model_name)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "traditional_confusion_matrices.png",
    dpi=300
)

plt.show()

# =====================================
# 9. TRAINING TIME COMPARISON
# =====================================

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Training Time (seconds)"
)

plt.title("Training Time Comparison")
plt.xlabel("Model")
plt.ylabel("Training Time (seconds)")
plt.tight_layout()

plt.savefig(
    "traditional_training_time.png",
    dpi=300
)

plt.show()

print("\nAll traditional ML experiments completed!")
print("Results and visualizations have been saved.")