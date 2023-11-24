from torch import nn

class FeatureExtractor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(FeatureExtractor, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, input_array):
        h = self.linear1(input_array)
        y_pred = self.linear2(h)
        return y_pred
    

## Wrapper that feeds hyperparams into FSL model to facilitate tuning
    ## May not be necessary...
class FSLNetworkWrapper ():
    def __init__(self) -> None:
        pass

    
    ## Required method for sklearn
    def fit (self, X, y):

        return
    

    ## Required method for sklearn
    def predict (self, X):

        return self.model(X)
    