# Iris Classification using MLP From Scratch

A machine learning project that implements **Iris flower classification using a Multi-Layer Perceptron (MLP) from scratch with NumPy**. The main objective of this project is to understand how a neural network works internally by manually implementing the core components instead of using a high-level deep learning framework.

The project uses the **Iris dataset from the UCI Machine Learning Repository**, containing four input features — sepal length, sepal width, petal length, and petal width — to classify flowers into three species: Iris-setosa, Iris-versicolor, and Iris-virginica. The dataset contains 150 samples, with 120 samples used for training and 30 samples used for testing in the selected train-test split.

The complete machine learning process starts with loading the dataset using `ucimlrepo`, separating the input features and target labels, encoding the categorical target classes, splitting the data into training and testing sets, and applying `StandardScaler` for feature normalization. Scaling the features is important because neural networks are sensitive to differences in feature ranges.

The neural network is implemented manually using NumPy. The project initially evaluates a baseline architecture of **4-4-3**, where the four input neurons correspond to the four Iris features, the hidden layer contains four neurons, and the output layer contains three neurons corresponding to the three flower species. A second architecture of **4-8-3** is also evaluated by increasing the hidden layer to eight neurons. This allows the project to examine whether a larger hidden representation improves classification performance.

The neural network training process includes weight and bias initialization, forward propagation, activation functions, loss calculation, backpropagation, gradient calculation, and parameter updates. During forward propagation, the input features pass through the hidden layer and then the output layer to generate predictions. Backpropagation is used to calculate gradients of the network parameters with respect to the loss, and the parameters are updated using gradient-based optimization. The model is trained using mini-batches rather than processing the entire training dataset in a single update.

Hyperparameter experiments are performed to understand how training parameters affect model performance. Learning rates of **0.001, 0.01, and 0.1** are evaluated, producing accuracies of approximately **33.33%, 66.67%, and 96.67%**, respectively, in the corresponding experiment. Different batch sizes of **8, 16, and 32** are also evaluated to study their effect on training. Based on the experiments, the selected model configuration uses the **4-8-3 architecture, learning rate 0.1, batch size 16, and 100 training epochs**.

The trained model is evaluated using **accuracy, precision, recall, F1-score, and classification reports**. In addition to standard evaluation, the project performs error analysis by comparing predicted and actual test labels and examines training and testing performance to understand the model's generalization behavior.

The project also includes an **interactive prediction interface using `ipywidgets`**. Users can enter sepal length, sepal width, petal length, and petal width values, after which the input is scaled using the same preprocessing pipeline and passed through the trained MLP. The interface returns the predicted Iris species along with the corresponding class probabilities.

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
