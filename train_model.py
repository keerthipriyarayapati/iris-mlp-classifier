"""
Train NumPy Multi-Layer Perceptron (MLP) for Iris Classification
Strictly follows the data preprocessing, architecture (4-8-3),
and training configuration from Iris.ipynb.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Artifacts output directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

DATASET_CSV = os.path.join(ARTIFACTS_DIR, 'iris_dataset.csv')


def load_dataset():
    """
    Fetch the UCI Iris dataset (ID 53).
    Saves a local CSV copy in artifacts/ for offline reproducibility.
    """
    if os.path.exists(DATASET_CSV):
        print(f"Loading cached Iris dataset from {DATASET_CSV}...")
        df = pd.read_csv(DATASET_CSV)
        X = df[['sepal length', 'sepal width', 'petal length', 'petal width']]
        y = df[['class']]
        return X, y

    try:
        print("Fetching Iris dataset from UCI Machine Learning Repository (ID 53)...")
        from ucimlrepo import fetch_ucirepo
        iris = fetch_ucirepo(id=53)
        X = iris.data.features
        y = iris.data.targets
        
        # Save local cache
        full_df = pd.concat([X, y], axis=1)
        full_df.to_csv(DATASET_CSV, index=False)
        print(f"Saved dataset cache to {DATASET_CSV}")
        return X, y
    except Exception as e:
        print(f"UCI fetch failed ({e}), falling back to scikit-learn iris dataset...")
        from sklearn.datasets import load_iris
        sk_iris = load_iris(as_frame=True)
        X = sk_iris.data
        X.columns = ['sepal length', 'sepal width', 'petal length', 'petal width']
        species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
        y = pd.DataFrame({'class': sk_iris.target.map(species_map)})
        full_df = pd.concat([X, y], axis=1)
        full_df.to_csv(DATASET_CSV, index=False)
        return X, y


# --- Activations and Loss from Notebook ---

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def one_hot_encode(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / m


def backward_propagation(X, y_true, Z1, A1, A2, W2):
    m = X.shape[0]
    dZ2 = A2 - y_true
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2


def train_model(
    X_train,
    y_train,
    input_size=4,
    hidden_size=8,
    output_size=3,
    learning_rate=0.1,
    batch_size=16,
    epochs=100
):
    np.random.seed(42)

    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    loss_history = []
    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            Z1, A1, Z2, A2 = forward_propagation(X_batch, W1, b1, W2, b2)
            loss = cross_entropy_loss(y_batch, A2)
            epoch_loss += loss * X_batch.shape[0]

            dW1, db1, dW2, db2 = backward_propagation(
                X_batch, y_batch, Z1, A1, A2, W2
            )

            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        epoch_loss /= n_samples
        loss_history.append(epoch_loss)

    return W1, b1, W2, b2, loss_history


def main():
    X, y = load_dataset()
    print(f"Dataset shape: Features {X.shape}, Target {y.shape}")

    # Label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.iloc[:, 0])
    classes = list(label_encoder.classes_)
    print("Class mapping:", {i: c for i, c in enumerate(classes)})

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    # Standard scaling fitted on training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # One-hot encode targets
    y_train_onehot = one_hot_encode(np.array(y_train), num_classes=3)

    # Train final proposed architecture (4-8-3, lr=0.1, bs=16, epochs=100)
    print("Training final NumPy MLP model (4-8-3)...")
    W1, b1, W2, b2, loss_history = train_model(
        np.array(X_train_scaled),
        y_train_onehot,
        input_size=4,
        hidden_size=8,
        output_size=3,
        learning_rate=0.1,
        batch_size=16,
        epochs=100
    )

    # Evaluate on test set
    _, _, _, y_prob = forward_propagation(np.array(X_test_scaled), W1, b1, W2, b2)
    y_pred = np.argmax(y_prob, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    print("\n--- Test Set Evaluation Results ---")
    print(f"Accuracy : {acc * 100:.2f}% ({acc:.4f})")
    print(f"Precision: {prec * 100:.2f}% ({prec:.4f})")
    print(f"Recall   : {rec * 100:.2f}% ({rec:.4f})")
    print(f"F1-Score : {f1 * 100:.2f}% ({f1:.4f})")
    print(f"Evaluated on {len(y_test)} test samples")

    # Save weights as NPZ
    weights_path = os.path.join(ARTIFACTS_DIR, 'model_weights.npz')
    np.savez_compressed(
        weights_path,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2
    )
    print(f"Model weights saved to {weights_path}")

    # Save scaler and label encoder parameters as JSON
    scaler_params = {
        "feature_names": list(X.columns),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "classes": classes
    }
    scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler_params.json')
    with open(scaler_path, 'w', encoding='utf-8') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"Scaler parameters saved to {scaler_path}")

    # Save metrics and training configuration as JSON
    metrics_data = {
        "dataset": {
            "name": "UCI Iris Dataset",
            "samples": int(X.shape[0]),
            "features_count": int(X.shape[1]),
            "features": list(X.columns),
            "classes_count": len(classes),
            "classes": classes
        },
        "model": {
            "name": "Multi-Layer Perceptron (MLP)",
            "implementation": "NumPy (from scratch)",
            "input_neurons": 4,
            "hidden_neurons": 8,
            "output_neurons": 3,
            "architecture": "4-8-3",
            "hidden_activation": "ReLU",
            "output_activation": "Softmax"
        },
        "training": {
            "learning_rate": 0.1,
            "batch_size": 16,
            "epochs": 100,
            "optimization": "Mini-batch Gradient Descent",
            "random_seed": 42
        },
        "performance": {
            "accuracy": f"{acc * 100:.2f}%",
            "precision": f"{prec * 100:.2f}%",
            "recall": f"{rec * 100:.2f}%",
            "f1_score": f"{f1 * 100:.2f}%",
            "test_samples": len(y_test),
            "raw_accuracy": acc,
            "raw_precision": prec,
            "raw_recall": rec,
            "raw_f1_score": f1
        }
    }
    metrics_path = os.path.join(ARTIFACTS_DIR, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Verify default sample
    sample_raw = np.array([[5.1, 3.5, 1.4, 0.2]])
    sample_scaled = (sample_raw - np.array(scaler.mean_)) / np.array(scaler.scale_)
    _, _, _, sample_prob = forward_propagation(sample_scaled, W1, b1, W2, b2)
    pred_idx = int(np.argmax(sample_prob))
    pred_species = classes[pred_idx]
    pred_conf = sample_prob[0][pred_idx] * 100
    print(f"\nDefault input (5.1, 3.5, 1.4, 0.2) -> {pred_species} ({pred_conf:.2f}%)")
    for cls_name, prob in zip(classes, sample_prob[0]):
        print(f"  {cls_name}: {prob * 100:.2f}%")


if __name__ == '__main__':
    main()
