import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def load_data():
    # Load dataset directly from URL
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)

    # Basic preprocessing
    df = df.dropna()              # remove missing values
    df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

    # Split features and target
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train():
    # Load dataset
    X_train, X_test, y_train, y_test = load_data()

    # Train a simple regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate using R² score as "accuracy"
    accuracy = r2_score(y_test, y_pred)

    # Save metrics.json
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": float(accuracy)}, f, indent=2)

    print(f"Training complete! R2 Score (accuracy): {accuracy:.4f}")

if __name__ == "__main__":
    train()