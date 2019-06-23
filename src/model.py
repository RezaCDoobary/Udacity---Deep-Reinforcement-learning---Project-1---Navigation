import torch
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers  = [64,64], drop_p = 0.3, dueling = False):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (array): Hidden number of nodes in each layer
            drop_p (float [0-1]) : Probability of dropping nodes (implementation of dropout)
            dueling (boolean) : If true, network is dueling network, otherwise false.
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        # Add the output layer
        self.output = nn.Linear(hidden_layers[-1], action_size)

        # Create a with_dualing instance parameter
        self.with_dueling = dueling

        # Add a value function approximator - to be used if dueling
        self.state_value = nn.Linear(hidden_layers[-1], 1)
        
        # dropout parameter added in case of concentration on a subset of nodes in NN.
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, input):
        """Build a network that maps state -> action values.
        
        Params
        ======
            input (Tensor[torch.Variable]): Input tensor in PyTorch model
        """
        for linear in self.hidden_layers:
            input = F.relu(linear(input))
            input = self.dropout(input)
        
        if self.with_dueling:
            advantage_function = self.output(input) 
            output =  self.state_value(input) + (advantage_function - torch.mean(advantage_function))
        else:
            output = self.output(input)
        return output