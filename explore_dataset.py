import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("email.csv")

# Clean category labels
df["Category"] = (
    df["Category"].astype(str).str.strip().str.lower()
)

# Keep only valid categories
df = df[df["Category"].isin(["ham", "spam"])].copy()

# Remove missing messages
df = df.dropna(subset=["Message"])

# Convert messages to strings
df["Message"] = df["Message"].astype(str)

# Remove empty messages
df = df[df["Message"].str.strip() != ""].copy()

print("\nCLEANED DATASET SHAPE:")
print(df.shape)

print("\nCLEANED CATEGORY DISTRIBUTION:")
print(df["Category"].value_counts())

# Dataset information
print("\nCLEANED DATASET SHAPE")
print(df.shape)

print("\nCATEGORY DISTRIBUTION")
print(df["Category"].value_counts())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

print("\nDUPLICATE MESSAGES")
print(df["Message"].duplicated().sum())

# Message length statistics
df["Message_Length"] = df["Message"].str.len()

print("\nMESSAGE LENGTH STATISTICS")
print(df["Message_Length"].describe())

# Plot category distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Category",
    order=["ham", "spam"]
)

plt.title("Distribution of Spam and Ham Messages")
plt.xlabel("Category")
plt.ylabel("Number of Messages")
plt.tight_layout()

plt.savefig(
    "category_distribution.png",
    dpi=300
)

plt.show()