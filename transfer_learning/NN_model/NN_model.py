import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImprovedNN(nn.Module):
    def __init__(self, input_size, dropout_rate=0.2, hidden_layers = [256, 128, 64]):
        super(ImprovedNN, self).__init__()

        layerlist = []
        n_in = input_size

        for h in hidden_layers:
            layerlist += [nn.Linear(n_in, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout_rate)]
            n_in = h
        
        # Final Output Layer
        layerlist.append(nn.Linear(n_in, 1))
        self.network = nn.Sequential(*layerlist)

                
        # Better initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)
    