import json
import re
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# 1. File paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "lstm_spam_model.pth"
VECTORIZER_PATH = BASE_DIR / "lstm_vectorizer.pkl"
CONFIG_PATH = BASE_DIR / "lstm_model_config.json"

DEVICE = torch.device("cpu")


# 2. Load configuration
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)

PAD_ID = config["pad_id"]
UNK_ID = config["unk_id"]
MAX_LENGTH = config["max_length"]
VOCAB_SIZE = config["vocab_size"]
EMBEDDING_DIM = config["embedding_dim"]
HIDDEN_DIM = config["hidden_dim"]
THRESHOLD = config["threshold"]


# 3. Clean message text
def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " url ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# 4. Define the LSTM model
class SpamLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        hidden_dim=64
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=PAD_ID
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, x, lengths):

        embedded = self.embedding(x)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        output = self.dropout(hidden[-1])

        output = self.fc(output)

        return output.squeeze(1)


# 5. Load the correct vectorizer
with open(VECTORIZER_PATH, "rb") as file:
    vectorizer = pickle.load(file)

vocab = vectorizer.vocabulary_
analyzer = vectorizer.build_analyzer()


# 6. Load the trained model
model = SpamLSTM(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()


# 7. Convert a message into a numerical sequence
def message_to_tensor(message):

    message = clean_text(message)

    tokens = analyzer(message)

    sequence = [
        vocab[token] + 2 if token in vocab else UNK_ID
        for token in tokens
    ]

    sequence = sequence[:MAX_LENGTH]

    length = len(sequence)

    # Handle empty messages
    if length == 0:
        sequence = [UNK_ID]
        length = 1

    # Add padding
    sequence += [PAD_ID] * (MAX_LENGTH - length)

    message_tensor = torch.tensor(
        [sequence],
        dtype=torch.long
    )

    length_tensor = torch.tensor(
        [length],
        dtype=torch.long
    )

    return message_tensor, length_tensor


# 8. Predict spam or ham
def predict_spam(message):

    message_tensor, length_tensor = message_to_tensor(
        message
    )

    message_tensor = message_tensor.to(DEVICE)

    length_tensor = length_tensor.to(DEVICE)

    with torch.no_grad():

        logits = model(
            message_tensor,
            length_tensor
        )

        probability = torch.sigmoid(logits).item()

    prediction = (
        "SPAM"
        if probability >= THRESHOLD
        else "HAM"
    )

    print("\n" + "=" * 45)
    print("MESSAGE:", message)
    print("PREDICTION:", prediction)
    print(f"SPAM PROBABILITY: {probability:.2%}")
    print("=" * 45)


# 9. Interactive prediction
print("\nLSTM EMAIL SPAM CLASSIFIER")
print("-" * 45)
print("Enter an email or message to classify.")
print("Type 'exit' to close the program.")

while True:

    message = input("\nEnter a message: ")

    if message.strip().lower() == "exit":
        print("Program closed.")
        break

    if not message.strip():
        print("Please enter a message.")
        continue

    predict_spam(message)