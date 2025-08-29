# Disease Prediction Toolkit

A machine learning toolkit for disease prediction.  
It provides preprocessing, model training, evaluation, and visualization in a clean, reusable structure.

## Features
- Data preprocessing (handle missing values, scaling, splitting)
- Model training (Logistic Regression, Decision Tree, Random Forest)
- Evaluation metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC)
- Visualization (Confusion Matrix, ROC Curve)

## Installation
```bash
git clone https://github.com/yourusername/disease_prediction_toolkit.git
cd disease_prediction_toolkit
pip install -r requirements.txt

from src.preprocessing import load_and_preprocess
from src.models import train_logistic
from src.evaluation import evaluate_model

# Load & preprocess
X_train, X_test, y_train, y_test = load_and_preprocess("diabetes.csv")

# Train
model = train_logistic(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
evaluate_model("Logistic Regression", y_test, y_pred, model, X_test)



---

### 🔹 Where each part goes
- **Usage** → inside `README.md` (under a heading `## Usage`)  
- **License** → also in `README.md` (usually at the end)  

👉 If you want to be extra professional, you can also create a separate `LICENSE` file (GitHub can auto-generate it for you when you create the repo → Add file → Create new file → name it `LICENSE` → choose MIT).  

---

✅ So: Both **Usage** and **License** sections live in your **README.md** file, unless you add a separate `LICENSE` file.  

---

Do you want me to also create the **requirements.txt** content (all the libraries your project needs) so you can add that too?
