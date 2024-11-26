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
import wandb


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

    # Initialize WandB
    wandb.init(
        project="vae-ae-training",  # Change the project name as needed
        config=config,
        notes="Training autoencoder or variational autoencoder",
    )
    wandb.run.name = f"{args.model}_training"

    # Initialize datasets (using MNIST)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])  # Flatten images
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

    # Model initialization
    if args.model == 'ae':
        m_base = ModelAE(config['ae_input_dim'], config['ae_latent_dim'])
        m_head = ModelHead(config['ae_latent_dim'])
        model = FullModel(m_base, m_head)

        # Initialize Xs_distill (distilled dataset)
        Xs_distill = torch.randn(config['n_distill'], config['ae_input_dim'], requires_grad=True)  # Ensure gradients are enabled
    elif args.model == 'vae':
        m_base = ModelVAE(config['vae_input_dim'], config['vae_latent_dim'], config['vae_output_dim'])  # Use VAE model
        m_head = ModelHead(config['vae_latent_dim'])
        model = FullModel(m_base, m_head)

        # Initialize Xs_distill (distilled dataset)
        Xs_distill = torch.randn(config['n_distill'], config['vae_input_dim'], requires_grad=True)  # Ensure gradients are enabled

    # Optimizer
    optimizer = optim.Adam(list(m_base.parameters()) + list(m_head.parameters()) + [Xs_distill], lr=config['lr'])

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xs_tr, ys_tr in train_loader:
            ys_tr = ys_tr.long()  # Convert labels to long type for cross-entropy
            optimizer.zero_grad()

            # Forward pass through the VAE
            if args.model == 'vae':
                reconstructed_Xs, mu, logvar = m_base(xs_tr)
                vae_loss = VAE_loss(reconstructed_Xs, xs_tr, mu, logvar)

            else:  # AE loss
                reconstructed_Xs = m_base(xs_tr)
                vae_loss = AE_loss(m_base, xs_tr)

            # Classification loss
            output_head = model(xs_tr)
            classification_loss_value = classification_loss(output_head, ys_tr)

            # Calculate the penalizer and diversity loss
            grad_penalty = config['grad_penalty'] * penalizer_term(m_base, Xs_distill)
            diversity_loss = config['divers_loss'] * F.mse_loss(Xs_distill, Xs_distill.mean(dim=0).repeat(Xs_distill.size(0), 1))

            # Total loss
            loss = classification_loss_value + grad_penalty + vae_loss - diversity_loss
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

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "loss": total_loss,
            "test_accuracy": test_accuracy,
        })

        # Save intermediate results (e.g., filters or dataset visuals)
        if epoch % config['save_every'] == 0:
            # Example of saving model filters or the distilled dataset (e.g., saved as images)
            # Save a 4x4 grid of distilled dataset visualization
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))  # Create a 4x4 grid of subplots
            for i, ax in enumerate(axes.flat):
                if i < Xs_distill.size(0):  # Ensure we don't exceed the number of available images
                    ax.imshow(Xs_distill[i].detach().numpy().reshape(28, 28), cmap='gray')
                    ax.axis('off')  # Turn off axes for cleaner visualization
            plt.tight_layout()
            grid_path = f"{config['save_dir']}/images/Xs_distilled_epoch_{epoch + 1}.png"
            plt.savefig(grid_path)
            plt.close(fig)  # Close the figure to avoid memory issues
            
            # Log the 4x4 grid to WandB
            wandb.log({"Xs_distill_grid": wandb.Image(grid_path)})


            # Save model
            torch.save(model.state_dict(), f"{config['save_dir']}/models/model_{epoch}.pth")
        # Save the final model
        torch.save(model.state_dict(), f"{config['save_dir']}/models/model_final.pth")
        wandb.save(f"{config['save_dir']}/models/model_final.pth")


# Loss function for VAE
def VAE_loss(reconstructed, original, mu, logvar):
    reconstruction_loss = F.mse_loss(reconstructed, original, reduction='sum')  # Or use BCE loss
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='ae')  # option that takes a value
    parser.add_argument('-c', '--config', type=str, default='./configs/train_config.yaml')  # option that takes a value
    args = parser.parse_args()
    train(args, config_path=args.config)
