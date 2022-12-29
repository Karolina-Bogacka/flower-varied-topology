import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
# borrowed from Pytorch quickstart example
from flwr.common import Parameters, NDArrays

from hypernetwork import Hypernetwork


class Net(nn.Module):

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# borrowed from Pytorch quickstart example
def train(net, trainloader, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


class CNNHyper(nn.Module):
    def __init__(
            self, n_nodes, embedding_dim, in_channels=3, out_dim=10, n_kernels=16,
            hidden_dim=100,
            spec_norm=False, n_hidden=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels
        self.embeddings = nn.Embedding(num_embeddings=n_nodes,
                                       embedding_dim=embedding_dim)

        from torch.nn.utils import spectral_norm
        layers = [
            spectral_norm(
                nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(
                embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(
                    nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(
                    hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.c1_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, self.n_kernels)
        self.c2_weights = nn.Linear(hidden_dim, 2* self.n_kernels * self.n_kernels * 5
                                    * 5)
        self.c2_bias = nn.Linear(hidden_dim, 2* self.n_kernels)
        self.l1_weights = nn.Linear(hidden_dim, 120 * 2* self.n_kernels * 5 * 5)
        self.l1_bias = nn.Linear(hidden_dim, 120)
        self.l2_weights = nn.Linear(hidden_dim, 84 * 120)
        self.l2_bias = nn.Linear(hidden_dim, 84)

        if spec_norm:
            self.c1_weights = spectral_norm(self.c1_weights)
            self.c1_bias = spectral_norm(self.c1_bias)
            self.c2_weights = spectral_norm(self.c2_weights)
            self.c2_bias = spectral_norm(self.c2_bias)
            self.l1_weights = spectral_norm(self.l1_weights)
            self.l1_bias = spectral_norm(self.l1_bias)
            self.l2_weights = spectral_norm(self.l2_weights)
            self.l2_bias = spectral_norm(self.l2_bias)

    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = OrderedDict({
            "conv1.weight": self.c1_weights(features).view(self.n_kernels,
                                                           self.in_channels, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(2 * self.n_kernels,
                                                           self.n_kernels, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(120, 2* self.n_kernels * 5 * 5),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(84, 120),
            "fc2.bias": self.l2_bias(features).view(-1)
        })
        return weights


class CIFARHyper(Hypernetwork):

    def __init__(self, network, optim='sgd', embed_lr=None, lr=1e-2, wd=1e-3):
        self.net = network
        optimizers = {
            'sgd': torch.optim.SGD(
                [
                    {'params': [p for n, p in self.net.named_parameters() if
                                'embed' not in n]},
                    {'params': [p for n, p in self.net.named_parameters() if 'embed' in
                                n],
                     'lr': embed_lr}
                ], lr=lr, momentum=0.9, weight_decay=wd
            ),
            'adam': torch.optim.Adam(params=self.net.parameters(), lr=lr)
        }
        self.optimizer = optimizers[optim]

    def fit(self, weights_ins: NDArrays, weights_res: NDArrays):
        self.optimizer.zero_grad()
        self.net.train()
        # calculating delta theta
        delta_theta = OrderedDict(
            {k: torch.tensor(prev - final, requires_grad = True) for k, (prev, final) in enumerate(zip(
                weights_ins, weights_res))})
        torch_weights = [torch.tensor(w, requires_grad = True) for w in weights_ins]
        # calculating phi gradient
        hnet_grads = torch.autograd.grad(torch_weights,
                                         self.net.parameters(),
                                         grad_outputs=list(delta_theta.values()),
                                         allow_unused=True
                                         )

        # update hnet weights
        for p, g in zip(self.net.parameters(), hnet_grads):
            p.grad = g

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 50)
        self.optimizer.step()

    def predict(self, index: int) -> NDArrays:
        self.net.eval()
        return self.net(torch.tensor([index], dtype=torch.long))


# borrowed from Pytorch quickstart example
def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy
