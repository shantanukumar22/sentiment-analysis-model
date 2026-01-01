import json
from utils.dataset_loader import load_dataset

def train():
    # TODO: Load your dataset
    X_train, X_test, y_train, y_test = load_dataset()

    # TODO: Create and train your model
    # model = ...
    # model.fit(X_train, y_train)

    # TODO: Evaluate your model properly
    accuracy = 0.0  # Replace with your actual accuracy

    # Save metrics for validator
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": accuracy}, f)

if __name__ == "__main__":
    train()
