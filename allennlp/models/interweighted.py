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


class finalModel(nn.Module):
    def __init__(self, hidden_length, cls_number):
        super(finalModel, self).__init__()

        self.fc1 = nn.Linear(hidden_length, int(hidden_length / 2))
        self.fc2 = nn.Linear(int(hidden_length / 2), cls_number)

    def forward(self, feature):
        f1 = F.relu(self.fc1(feature))
        f2 = self.fc2(f1)

        return f2