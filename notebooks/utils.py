from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, 
                              recall_score, precision_score, f1_score, roc_auc_score)


def classification_model_measurements(y_train, y_t_pred, y_test, y_pred, y_t_proba=None, y_proba=None):
    """
    Comprehensive model evaluation metrics.
    
    Parameters:
    -----------
    y_train, y_test : Ground truth labels
    y_t_pred, y_pred : Predicted binary labels
    y_t_proba, y_proba : (Optional) Predicted probabilities for ROC-AUC calculation
    """
    
    print("="*50)
    print(f"{' MODEL EVALUATION METRICS ':*^50}")
    print("="*50)
    
    # Accuracy
    print(f"\n{'ACCURACY SCORE':^50}")
    print("-"*50)
    print(f"Training  => {accuracy_score(y_train, y_t_pred)*100:.2f}%")
    print(f"Testing   => {accuracy_score(y_test, y_pred)*100:.2f}%")

    # Recall (Sensitivity / True Positive Rate)
    print(f"\n{'RECALL SCORE (Sensitivity)':^50}")
    print("-"*50)
    print(f"Training  => {recall_score(y_train, y_t_pred)*100:.2f}%")
    print(f"Testing   => {recall_score(y_test, y_pred)*100:.2f}%")
    
    # Precision
    print(f"\n{'PRECISION SCORE':^50}")
    print("-"*50)
    print(f"Training  => {precision_score(y_train, y_t_pred, zero_division=0)*100:.2f}%")
    print(f"Testing   => {precision_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    
    # F1-Score
    print(f"\n{'F1-SCORE (Harmonic Mean)':^50}")
    print("-"*50)
    print(f"Training  => {f1_score(y_train, y_t_pred, zero_division=0):.3f}")
    print(f"Testing   => {f1_score(y_test, y_pred, zero_division=0):.3f}")
    
    # ROC-AUC (if probabilities provided)
    if y_t_proba is not None and y_proba is not None:
        print(f"\n{'ROC-AUC SCORE':^50}")
        print("-"*50)
        print(f"Training  => {roc_auc_score(y_train, y_t_proba):.3f}")
        print(f"Testing   => {roc_auc_score(y_test, y_proba):.3f}")

    # Confusion Matrix
    print(f"\n{'CONFUSION MATRIX':^50}")
    print("-"*50)
    print(f"Training:\n{confusion_matrix(y_train, y_t_pred)}")
    print(f"\nTesting:\n{confusion_matrix(y_test, y_pred)}")

    # Classification Report
    print(f"\n{'CLASSIFICATION REPORT':^50}")
    print("-"*50)
    print(f"Training:\n{classification_report(y_train, y_t_pred)}")
    print(f"Testing:\n{classification_report(y_test, y_pred)}")
    print("="*50)
