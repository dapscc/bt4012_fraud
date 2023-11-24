import torch
from torch.utils.data import Dataset

## Needs to be a FewShotDataset / torch Dataset with .get_labels; consider using WrapFewShotDataset?
class FSLDataset (Dataset):
    def __init__ (self, dataframe, label_idx = -1, transformation = None):
        self.dataframe = dataframe
        self.transformation = transformation
        self.label_idx = label_idx
        # print('Disclaimer: It is assumed that the label is the last column of the input dataframe... ...')

    def __len__ (self):
        return len(self.dataframe)
    
    ## Returns Tensor(features), int(label)
    def __getitem__ (self, index):
        relevant_row = self.dataframe.iloc[index, :].values
        features = relevant_row[ : self.label_idx]
        label = relevant_row[self.label_idx]

        if self.transformation != None:
            features = self.transformation(features)
        
        return torch.Tensor(features), int(label)
    
    ## Required to use EasyFSL methods; returns a list of dataset's labels (TBC)
    def get_labels (self):      
        ret_list = self.dataframe.iloc[ : , self.label_idx]

        return ret_list.tolist()