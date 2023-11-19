from sklearn import metrics
import matplotlib.pyplot as plt

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


def get_metrics(actual, predicted, pos_label = 'Yes', neg_label = 'No'):

    accuracy = metrics.accuracy_score(actual, predicted)
    precision = metrics.precision_score(actual, predicted, pos_label = pos_label)
    recall = metrics.recall_score(actual, predicted, pos_label = pos_label)
    specificity = metrics.recall_score(actual, predicted, pos_label= neg_label)
    f1_score = metrics.f1_score(actual, predicted, pos_label = pos_label)

    return accuracy, precision, recall, specificity, f1_score