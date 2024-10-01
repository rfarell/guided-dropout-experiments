import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from models.guided_dropout_model import GuidedDropoutModel
from models.regular_dropout_model import RegularDropoutModel
from models.no_dropout_model import NoDropoutModel

def compute_activation_statistics(model, device, data_loader):
    model.eval()
    activations = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            x = data.view(data.size(0), -1)
            # First layer activations
            if isinstance(model, GuidedDropoutModel):
                # For GuidedDropoutModel, need to access the first layer's output manually
                x = model.fc1(x)
                x = F.relu(x)
                x = x * model.strength_fc1  # Apply strength parameter
            else:
                x = F.relu(model.fc1(x))
            activations.append(x.cpu().numpy())
    activations = np.concatenate(activations, axis=0)  # Shape: [num_samples, num_nodes]
    # Compute mean activation, variance, and SNR for each node
    mean_activation = np.mean(activations, axis=0)
    variance_activation = np.var(activations, axis=0)
    snr = mean_activation / (np.sqrt(variance_activation) + 1e-8)  # Add small epsilon to avoid division by zero
    return mean_activation, variance_activation, snr

def plot_strengths(metrics_dict, metric_name):
    plt.figure(figsize=(10, 6))
    for model_name, strengths in metrics_dict.items():
        # Normalize strengths by dividing by the maximum value
        strengths_normalized = strengths / strengths.max()
        # Sort in descending order
        sorted_strengths = np.sort(strengths_normalized)[::-1]
        plt.plot(sorted_strengths, label=model_name)
    plt.xlabel('Node Index (sorted)')
    plt.ylabel(f'Normalized {metric_name}')
    plt.title(f'Sorted and Normalized Node Strengths Based on {metric_name}')
    plt.legend()
    plt.tight_layout()
    # Save the plot with a filename that reflects normalization
    filename = './plots/'+metric_name.lower().replace(' ', '_') + '_strengths.png'
    plt.savefig(filename)
    plt.show()

def main():
    # Set device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data (or a subset of training data)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Load models
    models = {}
    model_filenames = {
        'Guided Dropout': 'trained_models/model_guided.pth',
        'Regular Dropout': 'trained_models/model_regular.pth',
        'No Dropout': 'trained_models/model_none.pth'
    }

    # Initialize models and load state dicts
    guided_model = GuidedDropoutModel().to(device)
    guided_model.load_state_dict(torch.load(model_filenames['Guided Dropout'], map_location=device))
    models['Guided Dropout'] = guided_model

    regular_model = RegularDropoutModel().to(device)
    regular_model.load_state_dict(torch.load(model_filenames['Regular Dropout'], map_location=device))
    models['Regular Dropout'] = regular_model

    no_dropout_model = NoDropoutModel().to(device)
    no_dropout_model.load_state_dict(torch.load(model_filenames['No Dropout'], map_location=device))
    models['No Dropout'] = no_dropout_model

    # Compute activation statistics for each model
    mean_activations = {}
    variances = {}
    snrs = {}

    for model_name, model in models.items():
        print(f'Computing activation statistics for {model_name}')
        mean_act, var_act, snr_act = compute_activation_statistics(model, device, test_loader)
        mean_activations[model_name] = mean_act
        variances[model_name] = var_act
        snrs[model_name] = snr_act

    # Plotting with normalization
    plot_strengths(mean_activations, 'Mean Activation')
    plot_strengths(variances, 'Variance of Activations')
    plot_strengths(snrs, 'Signal-to-Noise Ratio (SNR)')

if __name__ == '__main__':
    main()