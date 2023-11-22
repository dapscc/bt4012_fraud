import torch
from torch import nn

## TODO: Review implementation
    ## Can probably use EasyFSL's class, but implement own feature extractor
    ## Possible options for backbone: ... (Maybe can make do without?)

class PrototypicalNetwork (nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = backbone

    ## Predict query labels using labelled support data
    def forward (self, support_data: torch.Tensor, support_labels: torch.Tensor, 
                 query_data: torch.Tensor) -> torch.Tensor:
        ## Extract features / embedding of support and query data (using backbone)
        z_support = self.backbone.forward(support_data)
        z_query = self.backbone.forward(query_data)

        ## Infer no. of unique classes from support set labels
        n_way = len(torch.unique(support_labels))

        ## Construct prototypes
            ## Prototype i = Mean of embeddings of all support data with label i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0) \
                    for label in range(n_way)
            ]
        )

        ## Compute euclidean distance from query data to prototypes, and classification scores
        dists = torch.cdist(z_query, z_proto)
        classification_scores = -dists ## Smaller distance -> Higher score

        return classification_scores ## To be compared to actual query labels
    

class DummyNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(DummyNetwork, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, input_array):
        h = self.linear1(input_array)
        y_pred = self.linear2(h)
        return y_pred