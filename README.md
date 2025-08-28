# Neural Network from Scratch in Java 🧠

> **Educational Project**: Implementing a neural network from scratch in Java to deeply understand how neural networks work step by step.


---

## 🎯 Goal

Build and train a **Neural Network** for classification — such as **MNIST** or any custom dataset — using **Java only** (no external ML libraries).

---

## ⚡ Features

* Manual implementation of all core components.
* Matrix operations (addition, multiplication, transpose).
* Activation functions: Sigmoid, ReLU, Softmax.
* Forward propagation.
* Backpropagation.
* Gradient Descent optimization.
* Track accuracy and loss during training.
* Optionally implement optimizers like **Momentum** or **Adam**.
* Support for multiple hidden layers.

---

## 📂 Project Structure

```
NN-From-Scratch-in-Java/
├─ src/
│  ├─ matrix/       # Matrix operations
│  ├─ nn/           # Neural network and layers
│  ├─ util/         # Utilities (data loading, helpers)
│  └─ App.java      # Main entry point
├─ data/            # Training and test data
├─ models/          # Save trained models
├─ diagrams/        # Architecture diagrams
├─ README.md        # Documentation
└─ .gitignore
```

---

## 🚀 How to Run

### 1. Setup

* Install **JDK 11** or newer.
* Place your dataset inside the `data/` folder.

### 2. Compile & Run (PowerShell/Windows)

```powershell
javac -d out $(Get-ChildItem -Recurse -Filter "*.java" | ForEach-Object { $_.FullName })
java -cp out App
```

### 3. Configuration

In `App.java`, you can configure:

* Number of layers and neurons per layer.
* Activation functions.
* Learning rate.
* Number of epochs and batch size.

---

## 📊 Training & Evaluation

* During training, **loss** and **accuracy** values are printed per epoch.
* Models can be saved into the `models/` directory for later use.
* Basic API available: `Model.save(path)` and `Model.load(path)`.

---

## 🔧 Possible Improvements

* Implement **Adam Optimizer**.
* Advanced weight initialization (Xavier / He).
* Regularization techniques (L2, Dropout).
* Deeper architectures (more hidden layers).
* Parallelize matrix operations using **multithreading**.

---

## 📈 Example Report

When tested on MNIST:

* Accuracy: \~85-90% (with a simple architecture).
* Limitations: slower training compared to frameworks like TensorFlow, harder to scale with large datasets.

Include a `report.md` file describing:

* Experiment setup.
* Loss curve across epochs.
* Strengths and limitations.

---

## 🖼️ Diagrams

In `diagrams/`, include visual representations of:

* Input layer (number of features).
* Hidden layers (neurons and activations).
* Output layer (number of classes).

---

## 📝 Suggested .gitignore

```
# IntelliJ IDEA
.idea/
*.iml

# Compiled files
/out/
/bin/

# Models and Data
/models/
/data/
```

---

## 📜 License

This project is released under the MIT License — free to use, modify, and share.

---

💡 **Note**: The main purpose of this project is **learning and understanding the fundamentals**, not achieving state-of-the-art accuracy.


# Mostafa
# Mostafa2



---

## 🎯 Goal

Build and train a **Neural Network** for classification — such as **MNIST** or any custom dataset — using **Java only** (no external ML libraries).

---

## ⚡ Features

* Manual implementation of all core components.
* Matrix operations (addition, multiplication, transpose).
* Activation functions: Sigmoid, ReLU, Softmax.
* Forward propagation.
* Backpropagation.
* Gradient Descent optimization.
* Track accuracy and loss during training.
* Optionally implement optimizers like **Momentum** or **Adam**.
* Support for multiple hidden layers.

---

## 📂 Project Structure

```
NN-From-Scratch-in-Java/
├─ src/
│  ├─ matrix/       # Matrix operations
│  ├─ nn/           # Neural network and layers
│  ├─ util/         # Utilities (data loading, helpers)
│  └─ App.java      # Main entry point
├─ data/            # Training and test data
├─ models/          # Save trained models
├─ diagrams/        # Architecture diagrams
├─ README.md        # Documentation
└─ .gitignore
```

---

## 🚀 How to Run

### 1. Setup

* Install **JDK 11** or newer.
* Place your dataset inside the `data/` folder.

### 2. Compile & Run (PowerShell/Windows)

```powershell
javac -d out $(Get-ChildItem -Recurse -Filter "*.java" | ForEach-Object { $_.FullName })
java -cp out App
```

### 3. Configuration

In `App.java`, you can configure:

* Number of layers and neurons per layer.
* Activation functions.
* Learning rate.
* Number of epochs and batch size.

---

## 📊 Training & Evaluation

* During training, **loss** and **accuracy** values are printed per epoch.
* Models can be saved into the `models/` directory for later use.
* Basic API available: `Model.save(path)` and `Model.load(path)`.

---

## 🔧 Possible Improvements

* Implement **Adam Optimizer**.
* Advanced weight initialization (Xavier / He).
* Regularization techniques (L2, Dropout).
* Deeper architectures (more hidden layers).
* Parallelize matrix operations using **multithreading**.

---

## 📈 Example Report

When tested on MNIST:

* Accuracy: \~85-90% (with a simple architecture).
* Limitations: slower training compared to frameworks like TensorFlow, harder to scale with large datasets.

Include a `report.md` file describing:

* Experiment setup.
* Loss curve across epochs.
* Strengths and limitations.

---

## 🖼️ Diagrams

In `diagrams/`, include visual representations of:

* Input layer (number of features).
* Hidden layers (neurons and activations).
* Output layer (number of classes).

---

## 📝 Suggested .gitignore

```
# IntelliJ IDEA
.idea/
*.iml

# Compiled files
/out/
/bin/

# Models and Data
/models/
/data/
```

---

## 📜 License

This project is released under the MIT License — free to use, modify, and share.

---

💡 **Note**: The main purpose of this project is **learning and understanding the fundamentals**, not achieving state-of-the-art accuracy.

# Mostafa
# Mostafa2



# MAHMOUD



# MAHMOUD

