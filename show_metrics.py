from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def show_metrics(actual, predicted, pos_label = 'Yes', neg_label = 'No'):
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.show()
    accuracy = metrics.accuracy_score(actual, predicted)
    precision = metrics.precision_score(actual, predicted, pos_label = pos_label)
    recall = metrics.recall_score(actual, predicted, pos_label = pos_label)
    specificity = metrics.recall_score(actual, predicted, pos_label= neg_label)
    f1_score = metrics.f1_score(actual, predicted, pos_label = pos_label)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Specificity: {specificity}")
    print(f"F1_score: {f1_score}")

    # Generate ROC curve and calculate AUC
    y_pred_classes = [1 if i>0.5 else 0 for i in predicted]
    fpr, tpr, thresholds = roc_curve(actual, y_pred_classes)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


def get_metrics(actual, predicted, pos_label = 'Yes', neg_label = 'No'):

    accuracy = metrics.accuracy_score(actual, predicted)
    precision = metrics.precision_score(actual, predicted, pos_label = pos_label)
    recall = metrics.recall_score(actual, predicted, pos_label = pos_label)
    specificity = metrics.recall_score(actual, predicted, pos_label= neg_label)
    f1_score = metrics.f1_score(actual, predicted, pos_label = pos_label)

    return accuracy, precision, recall, specificity, f1_score

### Ignore this (for Deep learning) 
def show_metrics_DL(actual, predicted, samp, pos_label = 1, neg_label = 0):
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.show()
    accuracy = metrics.accuracy_score(actual, predicted)
    precision = metrics.precision_score(actual, predicted, pos_label = pos_label)
    recall = metrics.recall_score(actual, predicted, pos_label = pos_label)
    specificity = metrics.recall_score(actual, predicted, pos_label= neg_label)
    f1_score = metrics.f1_score(actual, predicted, pos_label = pos_label)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Specificity: {specificity}")
    print(f"F1_score: {f1_score}")

    # Generate ROC curve and calculate AUC
    y_pred_classes = [1 if i>0.5 else 0 for i in predicted]
    fpr, tpr, thresholds = roc_curve(actual, y_pred_classes)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({samp})')
    plt.legend(loc='lower right')
    plt.show()