from sklearn import metrics
import matplotlib.pyplot as plt

def show_metrics(actual, predicted):
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.show()
    accuracy = metrics.accuracy_score(actual, predicted)
    precision = metrics.precision_score(actual, predicted, pos_label="Yes")
    recall = metrics.recall_score(actual, predicted, pos_label="Yes")
    specificity = metrics.recall_score(actual, predicted, pos_label="No")
    f1_score = metrics.f1_score(actual, predicted, pos_label="Yes")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Specificity: {specificity}")
    print(f"F1_score: {f1_score}")