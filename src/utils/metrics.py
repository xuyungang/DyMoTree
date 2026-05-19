import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Union

def calculate_fate_metrics(y_true_prob: Union[np.ndarray, list], 
                           y_pred_prob: Union[np.ndarray, list], 
                           threshold: float = 0.5) -> Dict[str, float]:
    """
    A generalized function to evaluate cell fate prediction performance.
    Replaces the hardcoded get_metric, get_metric1, and get_m_early.
    
    Args:
        y_true_prob: Ground truth probabilities (e.g., Weinreb_fate or fate_bias).
        y_pred_prob: Model predicted probabilities.
        threshold: Binarization threshold for classification metrics.
        
    Returns:
        A dictionary containing AUROC, ACC, F1, Pearson, and Spearman scores.
    """
    y_true_prob = np.array(y_true_prob)
    y_pred_prob = np.array(y_pred_prob)
    
    # Binarize probabilities for classification metrics
    truth_label = np.where(y_true_prob > threshold, 1, 0)
    predict_label = np.where(y_pred_prob > threshold, 1, 0)

    # Calculate metrics
    acc = accuracy_score(y_true=truth_label, y_pred=predict_label)
    f1 = f1_score(y_true=truth_label, y_pred=predict_label, average='macro')
    
    try:
        auroc = roc_auc_score(truth_label, y_pred_prob)
    except ValueError:
        # Handles edge cases where only one class is present in y_true
        auroc = float('nan') 

    pearson_corr, _ = pearsonr(y_true_prob, y_pred_prob)
    spearman_corr, _ = spearmanr(y_true_prob, y_pred_prob)

    return {
        "auroc": auroc,
        "acc": acc,
        "f1": f1,
        "pearson": pearson_corr,
        "spearman": spearman_corr
    }