import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.nn import functional as F
import argparse
from models.models import *
import yaml
import os

def train(args, config_path='./configs/train_config.yaml'):
    assert os.path.exists(config_path), f"Config file not found at {config_path}"
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], 'models'), exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], 'images'), exist_ok=True)
    with open(f"{config['save_dir']}/config.yaml", 'w') as yamlfile:
        data = yaml.dump(config, yamlfile)
        yamlfile.close()
    print('='*10, 'Training Config', '='*10)
    print(config)

    # Initialize datasets (using MNIST)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])  # Flatten images
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

    # Model initialization
    if args.model == 'ae':
        m_base = ModelAE(config['input_dim'], config['latent_dim'])
    elif args.model == 'vae':
        raise NotImplementedError("Model not implemented yet")
        # TODO: implement VAE model in models.model
    m_head = ModelHead(config['latent_dim'])
    model = FullModel(m_base, m_head)

    # Initialize Xs_distill (distilled dataset)
    Xs_distill = torch.randn(config['n_distill'], config['input_dim'])  # Random initialization for distilled dataset

    # Optimizer
    optimizer = optim.Adam(list(m_base.parameters()) + list(m_head.parameters()) + [Xs_distill], lr=config['lr'])

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xs_tr, ys_tr in train_loader:
            ys_tr = ys_tr.long()  # Convert labels to long type for cross entropy
            optimizer.zero_grad()

            # Forward pass
            output_head = model(xs_tr)
            classification_loss_value = classification_loss(output_head, ys_tr)

            # Calculate the penalizer and AE loss
            grad_penalty = config['grad_penalty'] * penalizer_term(model.base_model, Xs_distill)
            ae_loss = config['inner_loss'] * AE_loss(model.base_model, Xs_distill)
            diversity_loss = config['divers_loss'] * F.mse_loss(Xs_distill, Xs_distill.mean(dim=0).repeat(Xs_distill.size(0), 1))

            # Total loss
            loss = classification_loss_value + grad_penalty + ae_loss - diversity_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Evaluation on test data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xs_test, ys_test in test_loader:
                ys_test = ys_test.long()
                outputs = model(xs_test)
                _, predicted = torch.max(outputs.data, 1)
                total += ys_test.size(0)
                correct += (predicted == ys_test).sum().item()

        test_accuracy = correct / total * 100

        # Output statistics
        total_loss /= n_batches
        print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {total_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        # Save intermediate results (e.g., filters or dataset visuals)
        if epoch % config['save_every'] == 0:
            # Example of saving model filters or the distilled dataset (e.g., saved as images)
            # Saving Xs_distill (distilled dataset visualization)
            plt.imshow(Xs_distill[0].detach().numpy().reshape(28, 28), cmap='gray')
            plt.savefig(f"{config['save_dir']}/images/Xs_distilled_epoch_{epoch + 1}.png")

            # Save  model
            torch.save(model.state_dict(), f"{config['save_dir']}/models/model_{epoch}.pth")
        # Save the final model
        torch.save(model.state_dict(), f"{config['save_dir']}/models/model_final.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='ae')  # option that takes a value
    args = parser.parse_args()
    train(args, config_path='./configs/train_config.yaml')