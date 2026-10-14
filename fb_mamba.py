"""FB-Mamba encoder and its selective state-space components."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Selective State Space Model (Mamba core).
    
    Implements input-dependent SSM with selective scan.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Causal conv1d
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=d_conv, padding=d_conv - 1, 
            groups=self.d_inner
        )
        
        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, delta
        
        # A parameter (learnable, initialized with log spacing)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Input projection: split into main path and gate
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_main, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)
        
        # Causal conv1d
        x_main = x_main.transpose(1, 2)  # (B, d_inner, L)
        x_main = self.conv1d(x_main)[:, :, :seq_len]  # causal: trim to seq_len
        x_main = x_main.transpose(1, 2)  # (B, L, d_inner)
        x_main = F.silu(x_main)
        
        # Selective scan
        y = self.selective_scan(x_main)
        
        # Gate
        z = F.silu(z)
        output = y * z
        
        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output
    
    def selective_scan(self, x):
        """Selective scan with input-dependent parameters.
        
        Uses parallel prefix scan for efficiency.
        
        Args:
            x: (batch, seq_len, d_inner)
        Returns:
            (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        
        # Compute input-dependent B, C, delta
        x_proj = self.x_proj(x)  # (B, L, d_state*2 + 1)
        B = x_proj[:, :, :self.d_state]  # (B, L, d_state)
        C = x_proj[:, :, self.d_state:2*self.d_state]  # (B, L, d_state)
        delta = F.softplus(x_proj[:, :, -1])  # (B, L)
        
        # Discretize: A_bar = exp(delta * A)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        delta = delta.unsqueeze(-1)  # (B, L, 1)
        
        # Compute discretized A and B per timestep
        # A_bar: (B, L, d_inner, d_state)
        dA = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (B, L, d_inner, d_state)
        A_bar = torch.exp(dA)
        # B_bar * x: (B, L, d_inner, d_state)
        Bx = (delta.unsqueeze(-1) * B.unsqueeze(2)) * x.unsqueeze(-1)  # (B, L, d_inner, d_state)
        
        # Parallel prefix scan: h[t] = A_bar[t] * h[t-1] + Bx[t]
        # Use log-space cumsum for A, then compute via associative scan
        # For moderate sequence lengths, chunked approach is efficient
        
        # Simple but fast: chunk into segments and process  
        CHUNK = 16
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        all_h = []
        
        for start in range(0, seq_len, CHUNK):
            end = min(start + CHUNK, seq_len)
            chunk_len = end - start
            
            # Process chunk recurrently (short chunks are fast)
            chunk_h = []
            for t in range(start, end):
                h = A_bar[:, t] * h + Bx[:, t]
                chunk_h.append(h)
            all_h.extend(chunk_h)
        
        # Stack: (B, L, d_inner, d_state)
        H = torch.stack(all_h, dim=1)
        
        # Output: y = C * h + D * x
        # C: (B, L, d_state), H: (B, L, d_inner, d_state) -> (B, L, d_inner)
        y = torch.einsum('bls,blds->bld', C, H) + self.D.unsqueeze(0).unsqueeze(0) * x
        
        return y


class FBMambaBlock(nn.Module):
    """Forward-Backward Mamba block with adaptive fusion."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd_ssm = SelectiveSSM(d_model, d_state, d_conv, expand, dropout)
        self.bwd_ssm = SelectiveSSM(d_model, d_state, d_conv, expand, dropout)
        
        # Adaptive fusion
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        residual = x
        x_normed = self.norm(x)
        
        # Forward pass
        h_fwd = self.fwd_ssm(x_normed)
        
        # Backward pass: flip, process, flip back
        x_flipped = torch.flip(x_normed, dims=[1])
        h_bwd = self.bwd_ssm(x_flipped)
        h_bwd = torch.flip(h_bwd, dims=[1])
        
        # Adaptive fusion (Eq. 11-13 in paper)
        alpha = torch.sigmoid(self.fusion_proj(torch.cat([h_fwd, h_bwd], dim=-1)))
        h_fused = alpha * h_fwd + (1 - alpha) * h_bwd
        
        # Output with layer norm
        output = self.fusion_norm(self.out_proj(h_fused))
        
        return residual + output


class FBMambaEncoder(nn.Module):
    """Forward-Backward Mamba encoder (proposed FB-Mamba)."""
    def __init__(self, input_channels, d_model=256, d_state=16, n_layers=3, 
                 d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.output_channels = d_model
        
        self.input_proj = nn.Linear(input_channels, d_model)
        self.blocks = nn.ModuleList([
            FBMambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (batch, seq_len, channels)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Global average pooling
        x = x.mean(dim=1)
        return x
