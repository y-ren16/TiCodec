import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# import modules
# import commons
LRELU_SLOPE = 0.1

class Proposed(nn.Module):

    # def __init__(self, n_specs=128, token_num=10, E=128, n_layers=4):
    def __init__(self, n_specs=128, E=128):

        super().__init__()
        self.encoder = SpeakerEncoder(in_channels=n_specs, hidden_channels=64, out_channels=E)
        # self.tl = nn.ModuleList([TokenLayer(E=E, token_num=token_num) for _ in range(n_layers)])

    def forward(self, inputs):
        inputs = inputs.transpose(1, 3).squeeze(-1)
        input_lengths = torch.tensor(inputs.shape[2]).repeat(inputs.shape[0]).to(inputs.device)
        enc_out = self.encoder(inputs, input_lengths)
        s = enc_out.unsqueeze(1) # [B, 1, C]
        embed = torch.zeros_like(s) # [B, 1, C]
        for layer in self.tl:
            h = layer(s)
            s = s - h
            embed = embed + h
        return embed

class TokenLayer(nn.Module):
    '''
    inputs --- [N, 1, E]
    '''

    def __init__(self, E, token_num):

        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(token_num, E))
        self.attention = CrossAttention(query_dim=E, key_dim=E, num_units=E)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        codebook = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)
        token = self.attention(inputs, codebook)

        return token


class CrossAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units):

        super().__init__()
        self.num_units = num_units
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(1, 2))  # [N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=2)

        # out = score * V
        out = torch.matmul(scores, values)  # [N, T_q, num_units]

        return out

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

class SpeakerEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(hidden_channels, hidden_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(hidden_channels, out_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(LRELU_SLOPE),
        )
        self.fn = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Linear(hidden_channels, out_channels),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.BatchNorm1d(out_channels),
        )
    def forward(self, x, x_lengths):
        """
        x --- [B, in_channels, T]
        out -- [B, out_channels]
        """
        x_mask = torch.unsqueeze(sequence_mask(
            x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.conv(x) * x_mask
        x = torch.sum(x, dim=2) / torch.sum(x_mask, dim=2) # [B, out_channels]
        x = self.fn(x)
        return x