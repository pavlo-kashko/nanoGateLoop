import torch
from torch import nn
from torch.nn import functional as F


class RecurrentScanGLRU(nn.Module):
    
    def __init__(self, config, eps = 1e-4, **kwargs):
        
        super(RecurrentScanGLRU, self).__init__(**kwargs)
        
        self.eps = eps
        self.d_h = config.d_h
        self.use_tied_gates = config.use_tied_gates
        self.n = 3 if self.use_tied_gates else 4
        self.input_activation = F.tanh
        self.hidden_activation = F.tanh
        self.gate_activation = F.sigmoid
        
    def forward(self, x):    
        """
        :param      x: float (batch_size, sequence_length, d_h * (3 if self.use_tied_gates is True else 4))
                    
        :return:    y: float (batch_size, sequence_length, d_h)
        """
        
        b, seq_len, _ = x.shape
                
        input_t = self.input_activation(x[:, :, :self.d_h])
        gates_t = self.gate_activation( x[:, :, self.d_h:])
        
        if self.use_tied_gates is True:
            input_gate_t, output_gate_t = torch.split(gates_t, self.d_h, dim=-1)
            forget_gate_t = 1 - input_gate_t
        else:
            input_gate_t, forget_gate_t, output_gate_t = torch.split(gates_t, self.d_h, dim=-1)
        
        kv = input_t * input_gate_t
        cum_prod_a = torch.cumprod(forget_gate_t, dim=1)
        y = cum_prod_a * torch.cumsum(kv / (cum_prod_a + self.eps), dim = 1)
                
        y = self.hidden_activation(y) * output_gate_t
        
        return y
    
    
class GatedLinearRNN(nn.Module):
    
    def __init__(self, config, **kwargs):
            
        super(GatedLinearRNN, self).__init__(**kwargs)
        
        self.n = 3 if config.use_tied_gates else 4
        self.proj_in  = nn.Linear(config.d_model, config.d_h * self.n)
        self.proj_out = nn.Linear(config.d_h, config.d_model)
        self.ln = nn.LayerNorm(config.d_model)
        
        self.model = RecurrentScanGLRU(config)
        
    def forward(self, x, carry=None):
        b, _, _ = x.shape
        x = self.proj_in(x)
        y = self.model(x)
        y = self.ln(self.proj_out(y))
        
        return y