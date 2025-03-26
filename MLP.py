import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MNIST dataset
train_dataset = MNIST(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root='./data', train=False, download=True, transform=ToTensor())

# Convert to numpy arrays and preprocess
x_train = train_dataset.data.numpy().reshape(-1, 784).astype('float32') / 255.0
y_train = train_dataset.targets.numpy()
x_test = test_dataset.data.numpy().reshape(-1, 784).astype('float32') / 255.0
y_test = test_dataset.targets.numpy()

# Convert to PyTorch tensors and move to device
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Configuration
neurons_list = [10, 20, 50, 100, 200, 500]
init_methods = ['random_normal', 'xavier', 'he', 'lecun', 'uniform']
activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh()
}
n_runs = 1
batch_size = 60
epochs = 10

# Initialize results dictionary (same structure as before)
results = {
    act_name: {
        init_method: {
            'neurons': neurons_list,
            'accuracy_mean': np.zeros(len(neurons_list)),
            'accuracy_std': np.zeros(len(neurons_list)),
            'train_time_mean': np.zeros(len(neurons_list)),
            'train_time_std': np.zeros(len(neurons_list)),
            'weight_stats': [{'W1_mean': 0, 'W1_std': 0, 'W2_mean': 0, 'W2_std': 0} for _ in neurons_list]
        }
        for init_method in init_methods
    }
    for act_name in activations.keys()
}

def get_initializer(init_method):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if init_method == 'random_normal':
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
            elif init_method == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif init_method == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif init_method == 'lecun':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
            elif init_method == 'uniform':
                nn.init.uniform_(m.weight, a=-0.05, b=0.05)
            nn.init.constant_(m.bias, 0.0)
    return init_weights

class MLP(nn.Module):
    def __init__(self, n_neurons, activation, init_method):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(784, n_neurons)
        self.layer2 = nn.Linear(n_neurons, n_neurons)
        self.output = nn.Linear(n_neurons, 10)
        self.activation = activation
        
        # Apply weight initialization
        self.apply(get_initializer(init_method))
        
        # Move model to device
        self.to(device)
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.output(x)
        return x

def train_model(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            # Move data to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def get_weight_stats(model):
    # Move weights to CPU for numpy conversion if they're on GPU
    W1 = model.layer1.weight.data.cpu().numpy()
    W2 = model.layer2.weight.data.cpu().numpy()
    return {
        'W1_mean': np.mean(W1),
        'W1_std': np.std(W1),
        'W2_mean': np.mean(W2),
        'W2_std': np.std(W2)
    }

# Main experiment loop
for act_name, activation in activations.items():
    print(f"\nTesting activation: {act_name}")
    for init_method in init_methods:
        print(f"\nInitialization method: {init_method}")
        for i, n_neurons in enumerate(tqdm(neurons_list)):
            accuracies = []
            train_times = []
            weight_stats_list = []
            
            for run in range(n_runs):
                # Create data loaders
                train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                # Create and train model
                model = MLP(n_neurons, activation, init_method)
                
                start_time = time.time()
                accuracy = train_model(model, train_loader, test_loader)
                train_time = time.time() - start_time
                
                weight_stats = get_weight_stats(model)
                
                accuracies.append(accuracy)
                train_times.append(train_time)
                weight_stats_list.append(weight_stats)
            
            # Store results
            results[act_name][init_method]['accuracy_mean'][i] = np.mean(accuracies)
            results[act_name][init_method]['accuracy_std'][i] = np.std(accuracies)
            results[act_name][init_method]['train_time_mean'][i] = np.mean(train_times)
            results[act_name][init_method]['train_time_std'][i] = np.std(train_times)
            
            # Average weight statistics across runs
            avg_weight_stats = {
                'W1_mean': np.mean([stat['W1_mean'] for stat in weight_stats_list]),
                'W1_std': np.mean([stat['W1_std'] for stat in weight_stats_list]),
                'W2_mean': np.mean([stat['W2_mean'] for stat in weight_stats_list]),
                'W2_std': np.mean([stat['W2_std'] for stat in weight_stats_list])
            }
            results[act_name][init_method]['weight_stats'][i] = avg_weight_stats

# Plotting functions (same as before)
def plot_results(results, activations, init_methods):
    """Plot all results with error bars"""
    colors = plt.cm.tab10.colors  # Color palette for different initialization methods
    
    # Plot Accuracy vs Neurons for each activation function
    for act_name in activations.keys():
        plt.figure(figsize=(12, 6))
        for i, init_method in enumerate(init_methods):
            data = results[act_name][init_method]
            plt.errorbar(
                data['neurons'], data['accuracy_mean'],
                yerr=data['accuracy_std'],
                marker='o', capsize=5, label=init_method,
                color=colors[i]
            )
        plt.xscale('log')
        plt.xlabel('Number of Neurons')
        plt.ylabel('Accuracy')
        title = f'Accuracy_vs_Neurons_for_{act_name}_Mean_Std_over_runs'
        plt.title(title.replace('_', ' '))  # Show nice title with spaces
        plt.legend()
        plt.grid()
        plt.savefig(f"{title}.png")  # Save with proper filename
        
        # Plot Training Time vs Neurons
        plt.figure(figsize=(12, 6))
        for i, init_method in enumerate(init_methods):
            data = results[act_name][init_method]
            plt.errorbar(
                data['neurons'], data['train_time_mean'],
                yerr=data['train_time_std'],
                marker='o', capsize=5, label=init_method,
                color=colors[i]
            )
        plt.xscale('log')
        plt.xlabel('Number of Neurons')
        plt.ylabel('Training Time (s)')
        title = f'Training_Time_vs_Neurons_for_{act_name}'
        plt.title(title.replace('_', ' '))
        plt.legend()
        plt.grid()
        plt.savefig(f"{title}.png")
    
    # Plot Weight Statistics
    for act_name in activations.keys():
        plt.figure(figsize=(15, 10))
        
        # W1 Mean
        plt.subplot(2, 2, 1)
        for i, init_method in enumerate(init_methods):
            data = results[act_name][init_method]
            w1_means = [stat['W1_mean'] for stat in data['weight_stats']]
            plt.plot(data['neurons'], w1_means, 'o-', label=init_method, color=colors[i])
        plt.xscale('log')
        plt.xlabel('Number of Neurons')
        plt.ylabel('Mean Weight Value')
        plt.title(f'W1 Mean for {act_name}')
        plt.legend()
        plt.grid()
        
        # W1 Std
        plt.subplot(2, 2, 2)
        for i, init_method in enumerate(init_methods):
            data = results[act_name][init_method]
            w1_stds = [stat['W1_std'] for stat in data['weight_stats']]
            plt.plot(data['neurons'], w1_stds, 'o-', label=init_method, color=colors[i])
        plt.xscale('log')
        plt.xlabel('Number of Neurons')
        plt.ylabel('Weight Std Dev')
        plt.title(f'W1 Standard Deviation for {act_name}')
        plt.legend()
        plt.grid()
        
        # W2 Mean
        plt.subplot(2, 2, 3)
        for i, init_method in enumerate(init_methods):
            data = results[act_name][init_method]
            w2_means = [stat['W2_mean'] for stat in data['weight_stats']]
            plt.plot(data['neurons'], w2_means, 'o-', label=init_method, color=colors[i])
        plt.xscale('log')
        plt.xlabel('Number of Neurons')
        plt.ylabel('Mean Weight Value')
        plt.title(f'W2 Mean for {act_name}')
        plt.legend()
        plt.grid()
        
        # W2 Std
        plt.subplot(2, 2, 4)
        for i, init_method in enumerate(init_methods):
            data = results[act_name][init_method]
            w2_stds = [stat['W2_std'] for stat in data['weight_stats']]
            plt.plot(data['neurons'], w2_stds, 'o-', label=init_method, color=colors[i])
        plt.xscale('log')
        plt.xlabel('Number of Neurons')
        plt.ylabel('Weight Std Dev')
        plt.title(f'W2 Standard Deviation for {act_name}')
        plt.legend()
        plt.grid()
        
        plt.tight_layout()
        plt.savefig(f'Weight_Statistics_for_{act_name}.png')
    
    # Max Accuracy per Initialization Method
    plt.figure(figsize=(15, 5))
    for i, act_name in enumerate(activations.keys()):
        plt.subplot(1, len(activations), i+1)
        max_accuracies = {
            init: np.max(data['accuracy_mean'])
            for init, data in results[act_name].items()
        }
        plt.bar(max_accuracies.keys(), max_accuracies.values(), color=colors[:len(init_methods)])
        plt.xlabel('Initialization Method')
        plt.ylabel('Max Mean Accuracy')
        plt.title(f'Max Accuracy for {act_name}')
        plt.grid()
    plt.tight_layout()
    plt.savefig('Max_Accuracy_per_Initialization_Method.png')

# Generate plots
plot_results(results, activations, init_methods)
