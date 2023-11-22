from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def forward_select(model, X_train, y_train):
    model.fit(X_train, y_train)
    ffs = SequentialFeatureSelector(model, k_features='best', forward=True, n_jobs=-1)
    ffs.fit(X_train, y_train) 
    features = list(ffs.k_feature_names_)
    print(f"Features selected: {features}")
    return features

def backward_select(model, X_train, y_train):
    model.fit(X_train, y_train)
    bfs = SequentialFeatureSelector(model, k_features='best', forward=False, n_jobs=-1)
    bfs.fit(X_train, y_train) 
    features = list(bfs.k_feature_names_)
    print(f"Features selected: {features}")
    return features

def rf_select(X_train, y_train):
    sel = SelectFromModel(RandomForestClassifier(), threshold= "0.5*mean")
    sel.fit(X_train, y_train)
    selected_feat= X_train.columns[(sel.get_support())]
    print(len(selected_feat))
    print(selected_feat)
    return selected_feat