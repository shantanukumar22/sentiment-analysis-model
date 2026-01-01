import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data():
    # SST2 dataset from GitHub (tab-separated)
    url = "https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/train.tsv"
    df = pd.read_csv(url, sep="\t", header=None, names=["sentence", "label"])

    # Drop rows with missing values
    df = df.dropna()

    X = df["sentence"]
    y = df["label"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train():
    # Load dataset
    X_train, X_test, y_train, y_test = load_data()

    # Text vectorizer
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train simple logistic regression classifier
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    # Predict
    y_pred = model.predict(X_test_vec)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Save accuracy to metrics.json
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": float(accuracy)}, f, indent=2)

    print(f"Training complete! Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train()