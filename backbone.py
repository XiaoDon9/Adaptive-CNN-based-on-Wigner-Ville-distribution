import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaptiveConv1d(nn.Module):
    """
    The adaptive convolutional layer has a kernel function composed of the input data itself
    Different output channels use different offsets:
        - The 0th output channel: Offset = 0
        - The first output channel: Offset = 1
        - The 2nd + output channel: The offset is a prime number arranged in ascending order
    When data is missing, it is truncated in a loop from the end of the data segment
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(AdaptiveConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.channel_offsets = self._generate_channel_offsets(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size))
    
    def _is_prime(self, n):
        """Determine whether a number is a prime number"""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def _generate_primes(self, count):
        """Generate a list of the specified number of prime numbers (from smallest to largest)"""
        primes = []
        num = 2
        while len(primes) < count:
            if self._is_prime(num):
                primes.append(num)
            num += 1
        return primes
    
    def _generate_channel_offsets(self, out_channels):
        """Generate the offsets of each output channel"""
        if out_channels <= 0:
            return []
        elif out_channels == 1:
            return [0]
        elif out_channels == 2:
            return [0, 1]
        else:
            prime_count = out_channels - 2
            primes = self._generate_primes(prime_count)
            return [0, 1] + primes
        
    def forward(self, x):
        batch_size, in_channels, length = x.shape
        
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding), mode='reflect')
            
        padded_length = x.shape[2]
        output_length = (padded_length - self.kernel_size) // self.stride + 1
        
        # unfolded: [batch_size, in_channels, output_length, kernel_size]
        unfolded = x.unfold(2, self.kernel_size, self.stride)
        
        output = torch.zeros(batch_size, self.out_channels, output_length, device=x.device, dtype=x.dtype)
        
        for out_ch in range(self.out_channels):
            channel_offset = self.channel_offsets[out_ch]
            
            pos_indices = torch.arange(output_length, device=x.device).unsqueeze(1) * self.stride  # [output_length, 1]
            kernel_offsets = torch.arange(self.kernel_size-1,-1,-1, device=x.device).unsqueeze(0)  # [1, kernel_size]
            source_positions = pos_indices - (self.kernel_size - 1 + channel_offset - kernel_offsets)
            valid_positions = source_positions % x.shape[2]
            
            # adaptive_kernels: [batch_size, in_channels, output_length, kernel_size]
            batch_idx = torch.arange(batch_size, device=x.device).view(-1, 1, 1, 1)
            channel_idx = torch.arange(in_channels, device=x.device).view(1, -1, 1, 1)
            pos_idx = valid_positions.view(1, 1, output_length, self.kernel_size)
            adaptive_kernels = x[batch_idx, channel_idx, pos_idx]
            
            # bias: [in_channels, kernel_size] -> [1, in_channels, 1, kernel_size]
            biased_kernels = adaptive_kernels + self.bias[out_ch:out_ch+1, :, None, :]
            conv_result = torch.sum(unfolded * biased_kernels, dim=(1, 3))
            output[:, out_ch, :] = conv_result
        return output


class CNN(nn.Module):
    def __init__(self, model_depth=4, dropout_rate=0.2, use_ADCNN_improve=False):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        self.use_ADCNN_improve = use_ADCNN_improve
        
        if use_ADCNN_improve:
            self.layers.append(AdaptiveConv1d(in_channels=1, out_channels=16, kernel_size=3))
            self.layers.append(nn.Dropout(dropout_rate))
        else:
            self.layers.append(nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3))
            self.layers.append(nn.BatchNorm1d(16))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
        
        in_channels = 16
        for i in range(1, model_depth):
            out_channels = in_channels * 2
            self.layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3))
            self.layers.append(nn.BatchNorm1d(out_channels))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels
            
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x
         
                    
class CNN_HardBottom(nn.Module):
    def __init__(self, num_classes, model_depth=4, init_method='kaiming_uniform', dropout_rate=0.2, use_ADCNN_improve=False):
        super(CNN_HardBottom, self).__init__()
    
        if isinstance(num_classes, tuple) and len(num_classes) == 2:
            num_fault_classes, num_condition_classes = num_classes
        else:
            raise ValueError("num_classes should be a tuple of (num_fault_classes, num_cond_outputs)")
        
        self.use_ADCNN_improve = use_ADCNN_improve
        self.init_method = init_method
        
        self.backbone = CNN(model_depth=model_depth, dropout_rate=dropout_rate, use_ADCNN_improve=use_ADCNN_improve)
        
        if model_depth == 1:
            feature_dim = 16
        else:
            feature_dim = 16 * (2 ** (model_depth - 1))

        self.fault_head = nn.Linear(feature_dim, num_fault_classes)
        self.cond_head = nn.Linear(feature_dim, num_condition_classes)
        
        self._initialize_weights(init_method)
        
    def forward(self, x):
        features = self.backbone(x)
        
        fault_preds = self.fault_head(features)
        cond_preds = self.cond_head(features)
        
        return fault_preds, cond_preds
    
    def loss_fun(self, preds, labels):
        fault_preds, cond_preds = preds
        fault_labels, cond_labels = labels
        
        fault_loss = F.cross_entropy(fault_preds, fault_labels)
        cond_loss = F.cross_entropy(cond_preds, cond_labels)
        
        return fault_loss, cond_loss
    
    def apply_improved_initialization(self):
        """Apply improved initialization if enabled"""
        if self.use_ADCNN_improve:
            print(f"Using ADCNN: {self.init_method}")
        else:
            print(f"Normalization method: {self.init_method}")
    
    def _initialize_weights(self, init_method):
        # map init names to functions
        init_funcs = {
            'kaiming_uniform': lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu'),
            'kaiming_normal':  lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu'),
            'xavier_uniform': lambda w: nn.init.xavier_uniform_(w),
            'xavier_normal':  lambda w: nn.init.xavier_normal_(w),
            'orthogonal':     lambda w: nn.init.orthogonal_(w),
            'normal':         lambda w: nn.init.normal_(w, mean=0, std=0.01)
        }
        init_fn = init_funcs.get(init_method)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                if init_fn:
                    init_fn(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
