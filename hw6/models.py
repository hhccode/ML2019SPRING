import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        self.layer_norm = nn.LayerNorm((100, embedding_dim))
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=hidden_dim*2, out_features=128),
            nn.LayerNorm(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )

        self.initialize_weights()
        
    def forward(self, x):
        x = self.layer_norm(x)
        x, _ = self.gru(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)

        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=127388, out_features=1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(True),
            nn.Dropout(0.35),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)
