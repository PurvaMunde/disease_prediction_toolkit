import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def plot_confusion_matrix(model, X_test, y_test, title="Confusion Matrix"):
    """Plot confusion matrix."""
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
    plt.title(title)
    plt.show()

def plot_roc_curve(model, X_test, y_test, title="ROC Curve"):
    """Plot ROC curve."""
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(title)
    plt.show()
