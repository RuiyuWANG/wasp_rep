import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

###################################### Models
# Define the Model_AE (Autoencoder)
class ModelAE(nn.Module):
    def __init__(self, dim_in, dim_out, weight_decay=1e-4, sigma=1.0):
        super(ModelAE, self).__init__()
        # Autoencoder architecture
        self.W = nn.Parameter(torch.empty(dim_in, dim_out).uniform_(-np.sqrt(6.0 / (dim_in + dim_out)),
                                                                    np.sqrt(6.0 / (dim_in + dim_out))))
        self.b = nn.Parameter(torch.zeros(dim_out))
        self.a = nn.Parameter(torch.zeros(dim_in))
        self.sigma = sigma
        self.weight_decay = weight_decay

    def forward(self, Xs):
        return F.relu(Xs @ self.W + self.b) @ self.W.T + self.a / self.sigma

# Define the Model_Head (Classification Head)
class ModelHead(nn.Module):
    def __init__(self, dim_in):
        super(ModelHead, self).__init__()
        # Classification layer
        self.W = nn.Parameter(
            torch.empty(dim_in, 10).uniform_(-np.sqrt(6.0 / (dim_in + 10)), np.sqrt(6.0 / (dim_in + 10))))
        self.b = nn.Parameter(torch.zeros(10))

    def forward(self, Xs):
        return F.relu(Xs) @ self.W + self.b

class FullModel(nn.Module):
    def __init__(self, base, head):
        super(FullModel, self).__init__()
        self.base_model = base
        self.head_model = head

    def forward(self, Xs):
        Zs = F.relu(Xs @ self.base_model.W + self.base_model.b)
        return Zs @ self.head_model.W + self.head_model.b


####################################### Losses and Accuracy
def classification_loss(ys_pred, ys_gt):
    return nn.CrossEntropyLoss()(ys_pred, ys_gt)

def AE_loss(model, Xs):
    B = Xs.size(1)
    XXs = model(Xs)
    delta_Xs = XXs - Xs / model.sigma
    return 0.5 * (torch.sum(delta_Xs ** 2) / B + model.weight_decay * torch.sum(model.W ** 2))

def accuracy(x, y):
    return torch.sum(torch.argmax(x, dim=0) == torch.argmax(y, dim=0)) / y.size(1)

###################################### others
#Initilization for dataset
def create_initial_Xs_distill(N=300):
    # Default N=300 as in your original code
    # Here you can replace it with real dataset loading logic (e.g., MNIST)
    # For demonstration, we create a random dataset (28*28 images, N samples)
    # Initialize the Xs_distill with random values (simulating synthetic data)
    Xs_distill = np.random.rand(28 * 28, N).astype(np.float32)
    Xs_distill /= 16.0
    return Xs_distill

# Define the penalizer function (smooth L1)
def penalizer_fn(u):
    return torch.sqrt(1.0 + u) - 1.0

# Function to calculate the gradient penalty term
def penalizer_term(m_base, Xs_distill):
    # Manually calculate the gradient for the penalizer term
    Xs_distill.requires_grad = True
    out = m_base(Xs_distill)
    grad_outputs = torch.ones_like(out)
    gradients = torch.autograd.grad(outputs=out, inputs=Xs_distill,
                                    grad_outputs=grad_outputs, create_graph=True)[0]
    grad_norm = torch.sum(gradients ** 2)
    return penalizer_fn(grad_norm)