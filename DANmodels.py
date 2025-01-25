import torch
import torch.nn as nn
import torch.nn.functional as F

class DAN(nn.Module):
    def __init__(self, embeddings, hidden_size):
        super(DAN, self).__init__()
        self.embedding = embeddings.get_initialized_embedding_layer(frozen=True)
        embedding_dim = embeddings.get_embedding_length()
        self.fc = nn.Linear(embedding_dim , hidden_size * 5)
        
        self.bn1 = nn.BatchNorm1d(hidden_size * 5)  
        self.fc1 = nn.Linear(hidden_size * 5 , hidden_size * 5)

        self.fc3 = nn.Linear(hidden_size * 5, 2)
        self.dropout = nn.Dropout(p=0.3)  
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, word_indices, lengths):
        word_indices = word_indices.long()
        embeddings = self.embedding(word_indices)  
        # print("Embeddings Shape:", embeddings.shape)

        mask = torch.arange(embeddings.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()  
        # print("Mask Example:", mask[0].shape)  
        
        embeddings = embeddings * mask  
        sum_embeddings = embeddings.sum(dim=1)  
        avg_embeddings = sum_embeddings / lengths.unsqueeze(1).float()  

        x = self.fc(avg_embeddings)
        x = self.bn1(x)
        x = F.rrelu(self.fc1(x))

        x = self.bn1(x)
        x = F.elu(self.fc1(x))
        x = F.dropout(x)
        
        x = self.fc3(x)
        return self.log_softmax(x)

