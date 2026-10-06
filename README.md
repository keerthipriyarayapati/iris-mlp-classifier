# Iris-mlp-classifier

A machine learning project that implements **Iris flower classification using a Multi-Layer Perceptron (MLP) from scratch with NumPy** to understand the internal working of neural networks without relying on high-level deep learning frameworks.

The project uses the **UCI Iris dataset** with four features — sepal length, sepal width, petal length, and petal width — to classify three species: Iris-setosa, Iris-versicolor, and Iris-virginica. The data is encoded, split into **120 training and 30 testing samples**, and standardized using `StandardScaler`.

Two architectures, **4-4-3** and **4-8-3**, are evaluated. The MLP manually implements weight initialization, forward propagation, activation functions, loss calculation, backpropagation, gradient updates, and mini-batch training.

Hyperparameter experiments evaluate learning rates **0.001, 0.01, and 0.1**, along with batch sizes **8, 16, and 32**. The selected configuration is **4-8-3 architecture, learning rate 0.1, batch size 16, and 100 epochs**.

Model performance is evaluated using **accuracy, precision, recall, F1-score, and classification reports**, along with error and generalization analysis. The project also includes an **interactive `ipywidgets` interface** for predicting Iris species and displaying class probabilities from user-provided measurements.

### Model Architecture

```text
                  ┌─────────────────┐
                  │   Input Layer   │
                  │    4 Neurons    │
                  │                 │
                  │ Sepal Length    │
                  │ Sepal Width     │
                  │ Petal Length    │
                  │ Petal Width     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Hidden Layer   │
                  │    8 Neurons    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Output Layer   │
                  │    3 Neurons    │
                  │                 │
                  │ Setosa          │
                  │ Versicolor      │
                  │ Virginica       │
                  └─────────────────┘
```

The final selected architecture is:

```text
4 → 8 → 3
```

where `4` represents the input features, `8` represents the hidden neurons, and `3` represents the output classes.

### Technology Stack

- **Python** – Programming language
- **NumPy** – Neural network implementation and numerical computation
- **Pandas** – Data processing
- **Scikit-learn** – Data preprocessing and evaluation utilities
- **Matplotlib** – Visualization
- **Seaborn** – Visualization
- **ucimlrepo** – UCI dataset access
- **ipywidgets** – Interactive prediction interface
- **Google Colab** – Development environment

### Installation and Usage

Clone the repository:

```bash
git clone https://github.com/keerthipriyarayapati/iris-mlp-classifier.git
```

Navigate to the project directory:

```bash
cd iris-mlp-classifier
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook Iris.ipynb
```

The notebook can also be opened and executed using Google Colab.

### Key Learning Outcomes

This project provides practical understanding of neural network architecture, Multi-Layer Perceptrons, forward propagation, backpropagation, gradient-based optimization, weight and bias updates, feature scaling, mini-batch training, hyperparameter experimentation, classification metrics, model evaluation, error analysis, generalization, and interactive machine learning prediction.

The main learning objective is to understand the internal working of a neural network and how its parameters are updated during training rather than treating the model as a black box.

### Future Improvements

Future improvements could include systematic hyperparameter search, additional activation functions, alternative optimization algorithms, cross-validation, confusion matrix visualization, training and validation curves, comparison with Scikit-learn's MLP implementation, and deployment of the model through a web-based prediction interface.
