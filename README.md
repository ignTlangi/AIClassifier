# 📈 AI Stock Classifier

## 🚀 Project Overview
This project implements three machine learning models to predict whether a financial stock should be purchased based on historical data:

- 🧬 **Genetic Programming (GP) Classifier** (Java)
- 🤖 **Multi-Layer Perceptron (MLP)** (Python, called from Java)
- 🌳 **Decision Tree (J48, Weka)** (Java)

The system is designed for reproducibility, clear results, and easy extensibility.

---

## 🗂️ Project Structure
```
AIStockClassifier/
├── data/               # Data files
│   ├── BTC_train.csv
│   └── BTC_test.csv
├── python/            # Python MLP implementation
│   └── MLP.py
├── results/           # Results output
├── src/              # Source code
│   ├── classifiers/  # Classifier implementations
│   │   ├── GPClassifier.java
│   │   ├── J48Classifier.java
│   │   └── MLPClassifier.java
│   ├── models/       # Model classes
│   └── utils/        # Utility classes
├── target/           # Maven build output
├── .vscode/          # VS Code settings
├── pom.xml           # Maven configuration
├── run.bat           # Windows run script
└── README.md         # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. **Prerequisites**
- Java 11 or higher
- Python 3.7+
- Maven (for dependency management)

### 2. **Install Python Dependencies**
```bash
pip install numpy pandas scikit-learn
```

### 3. **Build and Run**
#### Windows
Simply double-click `run.bat` or run it from the command line:
```bash
run.bat
```

#### Manual Build and Run
```bash
# Build the project
mvn clean install

# Run the application
java -cp target/stock-classifier-1.0-SNAPSHOT.jar StockClassifier
```

---

## 📝 Usage Guide

1. **Choose Classifier**
   - `1` for Genetic Programming (GP)
   - `2` for Multi-Layer Perceptron (MLP)
   - `3` for J48 Decision Tree
   - `4` to exit

2. **Enter Seed Value** (for GP and J48)
   - Enter an integer for reproducible results
   - The seed is used for randomization in training

3. **View Results**
   - Results are printed in the terminal
   - Detailed results are saved to the `results/` directory as CSV files
   - For J48, the decision tree structure is also displayed
   - Statistical significance test results are shown when comparing GP and MLP results

---

## 📊 Output Example

```
Model Evaluation Results:
-------------------------
Confusion Matrix:
True Positives (TP): 503
False Positives (FP): 24
True Negatives (TN): 446
False Negatives (FN): 25

Accuracy: 95.0902%
(BUY Precision): 0.9545
Recall: 0.9527
F1 Score: 0.9536

Wilcoxon Signed-Rank Test Results:
--------------------------------
p-value: 0.023456
Interpretation:
There is a statistically significant difference between the classifiers (p < 0.05)
```

CSV output includes all metrics and the confusion matrix for both training and test sets.

---

## 🔁 Reproducibility
- All classifiers support seed values for reproducible results
- Using the same seed and data will produce the same results
- Results are saved with timestamps for easy comparison
- Statistical tests ensure reliable comparison between classifiers

---

## 🛠️ Project Features
- Three different machine learning approaches
- Consistent evaluation metrics across all classifiers
- Automatic data preprocessing
- Detailed results and visualization
- Easy to use interface
- Statistical significance testing between classifiers

---

## 📚 Dependencies
- Weka 3.8.6 (for J48 Decision Tree)
- scikit-learn (for MLP)
- NumPy and Pandas (for data processing)
- JSON (for MLP communication)
- Apache Commons Math (for statistical tests)

---

## 📅 Future Improvements
- [ ] Add more classifiers
- [ ] Implement cross-validation
- [ ] Add feature importance analysis
- [ ] Improve visualization of results
- [ ] Add more statistical tests for classifier comparison

---

*For any issues or questions, please contact the project team.*
