
# 🔢 Linear Regression using PyTorch (Single Input Feature)

This project demonstrates how to build, train, and evaluate a simple **Linear Regression Model** using **PyTorch** with a single input feature: `YearsExperience`. The model learns to predict an employee's salary based on their years of experience.

---

## 🚀 Project Overview

- ✅ Model: `y = w * x + b`
- ✅ Framework: PyTorch
- ✅ Loss Function: Mean Squared Error (MSE)
- ✅ Optimizer: Stochastic Gradient Descent (SGD)
- ✅ Dataset: Synthetically generated and saved as `salary_data.csv`
- ✅ Visualization: Actual vs Predicted salary using `matplotlib`

---

## 📂 Dataset (`salary_data.csv`)

| YearsExperience | Salary   |
|------------------|----------|
| 1.1              | 47,000   |
| 2.5              | 53,000   |
| ...              | ...      |

The dataset contains:
- `YearsExperience`: Number of years of professional experience.
- `Salary`: Corresponding salary in dollars (target).

---

## 🏗️ Model Architecture

```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # Input: 1 feature → Output: 1 prediction

    def forward(self, x):
        return self.linear(x)
````

---

## 🧠 Training Logic

1. Load and normalize the dataset using `StandardScaler`.
2. Convert data to `torch.Tensor` with `float32` dtype.
3. Define model, loss (`nn.MSELoss`), and optimizer (`SGD`).
4. Train the model over multiple epochs.
5. Visualize predicted vs actual salary.

---

## 📊 Output Visualization

* Shows **how well the model fits** the training data.
* Compares actual salary points with the model's predictions.

<p align="center">
  <img src="prediction_plot.png" width="600" alt="Actual vs Predicted">
</p>

---

## 📦 Dependencies

* Python ≥ 3.8
* PyTorch
* pandas
* matplotlib
* scikit-learn

Install them using:

```bash
pip install torch pandas matplotlib scikit-learn
```

---

## 📁 Files

* `salary_data.csv` → Synthetic dataset
* `linear_regression.py` → Complete training and evaluation script
* `prediction_plot.png` → Saved output plot

---

## 🤖 Future Improvements

* Extend to **multi-feature regression** (e.g., add `EducationYears`, `Age`)
* Integrate with **GUI** using Streamlit or Tkinter
* Export trained model using TorchScript or ONNX

---

