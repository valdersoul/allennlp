import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

class interWeighted(nn.Module):
    def __init__(self, hidden_length):
        super(interWeighted, self).__init__()

        self.fc1 = nn.Linear(hidden_length, int(hidden_length / 2))
        self.fc2 = nn.Linear(int(hidden_length / 2), 1)

    def forward(self, setences, questions):
        q = torch.mean(questions, 1).unsqueeze(1)
        feature = setences * q

        f1 = F.tanh(self.fc1(feature))
        f2 = self.fc2(f1)

        return f2.squeeze()

class posAvg(nn.Module):
    def __init__(self, emb_size):
        super(posAvg, self).__init__()

        self.fc1 = nn.Linear(emb_size, 1)
    
    def forward(self, pos):
        weights = self.fc1(pos)
        return weights

class finalModel(nn.Module):
    def __init__(self, hidden_length, cls_number):
        super(finalModel, self).__init__()

        self.fc1 = nn.Linear(hidden_length, int(hidden_length / 2))
        self.fc2 = nn.Linear(int(hidden_length / 2), cls_number)

    def forward(self, feature):
        f1 = F.relu(self.fc1(feature))
        f2 = self.fc2(f1)

        return f2

class fusionNet(nn.Module):
    def __init__(self, in_size, out_size):
        super(fusionNet, self).__init__()
        self.fusion = nn.Linear(in_size, out_size)
        self.score = nn.Linear(in_size, 1)

    def forward(self, input, fusion):
        gate = F.sigmoid(self.score(fusion))
        fusion_part = F.tanh(self.fusion(fusion))
        return gate * fusion_part + (1 - gate) * input
