from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule

def undersample(X_train, y_train):
    undersampled_data = RandomUnderSampler(sampling_strategy='majority')
    X_under, y_under = undersampled_data.fit_resample(X_train, y_train)
    return X_under, y_under

def oversample(X_train, y_train):
    oversampled_data = RandomOverSampler(sampling_strategy='minority')
    X_over, y_over = oversampled_data.fit_resample(X_train, y_train)
    return X_over, y_over

def smote(X_train, y_train):
    smote_data = SMOTE()
    X_smote, y_smote = smote_data.fit_resample(X_train, y_train)
    return X_smote, y_smote

def ncr(X_train, y_train):
    undersample = NeighbourhoodCleaningRule(n_neighbors=3, threshold_cleaning=0.5)
    y_copy = y_train.copy()
    y_copy = y_copy.replace("Yes", 1)
    y_copy = y_copy.replace("No", 0)
    X_ncr, y_ncr = undersample.fit_resample(X_train, y_copy)
    # y_ncr = y_ncr.replace("Yes", 1)
    # y_ncr = y_ncr.replace("No", 0)
    # y_ncr = y_ncr.replace(1, "Yes")
    # y_ncr = y_ncr.replace(0, "No")
    return X_ncr, y_ncr