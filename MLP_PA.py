import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Verificar disponibilidade da CUDA e definir dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ====================== Funções de Rewiring PA ==============================
def rewiring_torch(weights, seed=None):
    dimensions = weights.shape
    if seed:
        torch.manual_seed(seed)
    
    st = torch.zeros(dimensions[1], device=weights.device)
    for neuron in range(1, dimensions[0]):
        st = st + weights[neuron - 1]
        P = st + torch.abs(torch.min(st)) + 1
        P = P / torch.sum(P)
        
        # Criar distribuição multinomial para amostragem
        targets = torch.multinomial(P, dimensions[1], replacement=False)
        
        edges_to_rewire = torch.argsort(weights[neuron])
        weights[neuron, targets] = weights[neuron, edges_to_rewire]
    return weights

def PA_rewiring_torch(weights, seed=None):
    if weights.ndim < 2:
        raise ValueError("Apenas tensores 2D ou superiores são suportados")
    
    original_shape = weights.shape
    output_neurons = weights.shape[0]
    input_neurons = weights.numel() // output_neurons
    
    weights_out = weights.view((output_neurons, input_neurons)).clone()
    weights_out = rewiring_torch(weights_out, seed)  # Rewiring nas linhas
    
    # Rewiring nas colunas (transpor, rewiring, transpor de volta)
    weights_out_transposed = weights_out.t()
    weights_out_transposed = rewiring_torch(weights_out_transposed, seed)
    weights_out = weights_out_transposed.t()
    
    return weights_out.view(original_shape)
# =============================================================================

# Carregar dataset MNIST
train_dataset = MNIST(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root='./data', train=False, download=True, transform=ToTensor())

# Converter para arrays numpy e pré-processar
x_train = train_dataset.data.numpy().reshape(-1, 784).astype('float32') / 255.0
y_train = train_dataset.targets.numpy()
x_test = test_dataset.data.numpy().reshape(-1, 784).astype('float32') / 255.0
y_test = test_dataset.targets.numpy()

# Converter para tensores PyTorch e mover para o dispositivo
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Configuração
neurons_list = [1, 2, 5, 10, 20, 50, 100, 200, 500]
init_methods = ['uniform']
activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh()
}
n_runs = 10
batch_size = 60
epochs = 1

# Dicionário de resultados
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
        
        # Aplicar inicialização
        self.apply(get_initializer(init_method))
        
        # Aplicar rewiring uma vez após inicialização
        #with torch.no_grad():
        #    self.layer1.weight.data = PA_rewiring_torch(self.layer1.weight.data)
        #    self.layer2.weight.data = PA_rewiring_torch(self.layer2.weight.data)
        
        self.to(device)
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.output(x)
        return x

def train_model(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Loop de treinamento
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Avaliação
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
    # Mover pesos para CPU para conversão numpy se estiverem na GPU
    W1 = model.layer1.weight.data.cpu().numpy()
    W2 = model.layer2.weight.data.cpu().numpy()
    return {
        'W1_mean': np.mean(W1),
        'W1_std': np.std(W1),
        'W2_mean': np.mean(W2),
        'W2_std': np.std(W2)
    }

# Loop principal do experimento
for act_name, activation in activations.items():
    print(f"\nTestando ativação: {act_name}")
    for init_method in init_methods:
        print(f"\nMétodo de inicialização: {init_method}")
        for i, n_neurons in enumerate(tqdm(neurons_list)):
            accuracies = []
            train_times = []
            weight_stats_list = []
            
            for run in range(n_runs):
                # Criar data loaders
                train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                # Criar e treinar modelo
                model = MLP(n_neurons, activation, init_method)
                
                start_time = time.time()
                accuracy = train_model(model, train_loader, test_loader)
                train_time = time.time() - start_time
                
                weight_stats = get_weight_stats(model)
                
                accuracies.append(accuracy)
                train_times.append(train_time)
                weight_stats_list.append(weight_stats)
            
            # Armazenar resultados
            results[act_name][init_method]['accuracy_mean'][i] = np.mean(accuracies)
            results[act_name][init_method]['accuracy_std'][i] = np.std(accuracies)
            results[act_name][init_method]['train_time_mean'][i] = np.mean(train_times)
            results[act_name][init_method]['train_time_std'][i] = np.std(train_times)
            
            # Estatísticas médias dos pesos entre as execuções
            avg_weight_stats = {
                'W1_mean': np.mean([stat['W1_mean'] for stat in weight_stats_list]),
                'W1_std': np.mean([stat['W1_std'] for stat in weight_stats_list]),
                'W2_mean': np.mean([stat['W2_mean'] for stat in weight_stats_list]),
                'W2_std': np.mean([stat['W2_std'] for stat in weight_stats_list])
            }
            results[act_name][init_method]['weight_stats'][i] = avg_weight_stats

# Função de plotagem (mesma de antes)
def plot_results(results, activations, init_methods):
    """Plotar todos os resultados com barras de erro"""
    colors = plt.cm.tab10.colors
    
    # Criar diretório de saída
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Plotar Acurácia vs Neurônios
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
        plt.xlabel('Número de Neurônios')
        plt.ylabel('Acurácia')
        title = f'Acurácia_vs_Neurônios_para_{act_name}_Média_Desvio'
        plt.title(title.replace('_', ' '))
        plt.legend()
        plt.grid()
        plt.savefig(f'plots/{title}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plotar Tempo de Treinamento vs Neurônios
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
        plt.xlabel('Número de Neurônios')
        plt.ylabel('Tempo de Treinamento (s)')
        title = f'Tempo_Treinamento_vs_Neurônios_para_{act_name}'
        plt.title(title.replace('_', ' '))
        plt.legend()
        plt.grid()
        plt.savefig(f'plots/{title}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Plotar Estatísticas dos Pesos
    for act_name in activations.keys():
        plt.figure(figsize=(15, 10))
        
        # W1 Média
        plt.subplot(2, 2, 1)
        for i, init_method in enumerate(init_methods):
            data = results[act_name][init_method]
            w1_means = [stat['W1_mean'] for stat in data['weight_stats']]
            plt.plot(data['neurons'], w1_means, 'o-', label=init_method, color=colors[i])
        plt.xscale('log')
        plt.xlabel('Número de Neurônios')
        plt.ylabel('Valor Médio do Peso')
        plt.title(f'W1 Média para {act_name}')
        plt.legend()
        plt.grid()
        
        # W1 Desvio Padrão
        plt.subplot(2, 2, 2)
        for i, init_method in enumerate(init_methods):
            data = results[act_name][init_method]
            w1_stds = [stat['W1_std'] for stat in data['weight_stats']]
            plt.plot(data['neurons'], w1_stds, 'o-', label=init_method, color=colors[i])
        plt.xscale('log')
        plt.xlabel('Número de Neurônios')
        plt.ylabel('Desvio Padrão do Peso')
        plt.title(f'W1 Desvio Padrão para {act_name}')
        plt.legend()
        plt.grid()
        
        # W2 Média
        plt.subplot(2, 2, 3)
        for i, init_method in enumerate(init_methods):
            data = results[act_name][init_method]
            w2_means = [stat['W2_mean'] for stat in data['weight_stats']]
            plt.plot(data['neurons'], w2_means, 'o-', label=init_method, color=colors[i])
        plt.xscale('log')
        plt.xlabel('Número de Neurônios')
        plt.ylabel('Valor Médio do Peso')
        plt.title(f'W2 Média para {act_name}')
        plt.legend()
        plt.grid()
        
        # W2 Desvio Padrão
        plt.subplot(2, 2, 4)
        for i, init_method in enumerate(init_methods):
            data = results[act_name][init_method]
            w2_stds = [stat['W2_std'] for stat in data['weight_stats']]
            plt.plot(data['neurons'], w2_stds, 'o-', label=init_method, color=colors[i])
        plt.xscale('log')
        plt.xlabel('Número de Neurônios')
        plt.ylabel('Desvio Padrão do Peso')
        plt.title(f'W2 Desvio Padrão para {act_name}')
        plt.legend()
        plt.grid()
        
        plt.tight_layout()
        plt.savefig(f'plots/Estatísticas_Pesos_para_{act_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Máxima Acurácia por Método de Inicialização
    plt.figure(figsize=(15, 5))
    for i, act_name in enumerate(activations.keys()):
        plt.subplot(1, len(activations), i+1)
        max_accuracies = {
            init: np.max(data['accuracy_mean'])
            for init, data in results[act_name].items()
        }
        plt.bar(max_accuracies.keys(), max_accuracies.values(), color=colors[:len(init_methods)])
        plt.xlabel('Método de Inicialização')
        plt.ylabel('Máxima Acurácia Média')
        plt.title(f'Máxima Acurácia para {act_name}')
        plt.grid()
    plt.tight_layout()
    plt.savefig('plots/Máxima_Acurácia_por_Método_Inicialização.png', dpi=300, bbox_inches='tight')
    plt.show()

# Gerar gráficos
plot_results(results, activations, init_methods)
