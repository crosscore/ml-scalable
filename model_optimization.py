import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# シンプルなニューラルネットワークの定義
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()

# fc1層の重みを30%プルーニング
prune.l1_unstructured(model.fc1, name='weight', amount=0.3)
print("プルーニング後のfc1の重み:", model.fc1.weight)