import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import dgl
from dgl.nn import GATConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from src.dlutils import *
from src.constants import *

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = input_dim ** -0.5

    def forward(self, x, return_attn = False):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out + x

        if return_attn:
            return out, attn
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # shape: [1, max_len, dim]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class CrossWindowTransformer(nn.Module):
    def __init__(self, embed_dim, nhead=4, num_layers=1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(embed_dim, nhead)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # [B, T, H] → [T, B, H]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [T, B, H] → [B, T, H]
        return x
    
class LSTMGenerator(nn.Module):
    def __init__(self, z_dim, lstm_hidden_dim, hidden_dim, output_dim, window_size = 5):
        super(LSTMGenerator, self).__init__()
        self.window_size = window_size
        self.lstm = nn.LSTM(z_dim, lstm_hidden_dim, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim // window_size),
        )

    def forward(self, z):
        out, _ = self.lstm(z) 
        out = self.decoder(out) 
        out = out.view(z.size(0), -1)
        return out.view(out.size(0), -1)

class STGAD(nn.Module):
    def __init__(self, input_dim, hidden_dim = 64, z_dim = 32, window_size=5, lstm_hidden_dim=64):
        super(STGAD, self).__init__()
        self.name = 'STGAD'
        self.lr = 0.0001
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.data_min = None
        self.data_max = None


        assert input_dim % window_size == 0, "input_dim % window_size"
        self.feature_dim = input_dim // window_size
        self.generator = LSTMGenerator(z_dim, lstm_hidden_dim, hidden_dim, input_dim, window_size=window_size)

        self.encoder = nn.Linear(self.feature_dim, hidden_dim)
        self.attn = SelfAttention(hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.cross_window_transformer = CrossWindowTransformer(embed_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def generate(self, batch_size, device):
        z = torch.randn(batch_size, self.window_size, self.z_dim).to(device)
        return self.generator(z)


    def generate_n(self, batch_size, device, n_samples: int):
        z = torch.randn(n_samples, batch_size, self.window_size, self.z_dim, device=device)
        z = z.view(n_samples * batch_size, self.window_size, self.z_dim)
        x = self.generator(z)
        return x.view(n_samples, batch_size, -1)

    def discriminate(self, x):
        batch_size = x.size(0)
        assert x.size(1) == self.input_dim

        x = x.view(batch_size, self.window_size, self.feature_dim)
        x_reshaped = x.reshape(-1, self.feature_dim)

        h = self.encoder(x_reshaped)
        h = F.leaky_relu(h, 0.2)
        h = h.view(batch_size, self.window_size, self.hidden_dim)

        h = self.attn(h)
        h = self.pos_encoding(h)
        h = self.cross_window_transformer(h)
        h = h.mean(dim=1)

        out = self.classifier(h)
        return out
    
    def discriminate_with_attn(self, x):
        batch_size = x.size(0)
        assert x.size(1) == self.input_dim

        x = x.view(batch_size, self.window_size, self.feature_dim)
        x_reshaped = x.reshape(-1, self.feature_dim)

        h = self.encoder(x_reshaped)
        h = F.leaky_relu(h, 0.2)
        h = h.view(batch_size, self.window_size, self.hidden_dim)

        h, attn_weights = self.attn(h, return_attn=True)
        h = self.pos_encoding(h)
        h = self.cross_window_transformer(h)

        h = h.mean(dim=1)
        out = self.classifier(h)
        return out, attn_weights

class STGAD_Standard(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, z_dim=32, window_size=5, lstm_hidden_dim=64):
        super().__init__()
        self.name = 'STGAD_Standard'
        self.lr = 0.0001
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.lstm_hidden_dim = lstm_hidden_dim
        assert input_dim % window_size == 0
        self.feature_dim = input_dim // window_size
        self.generator = LSTMGenerator(z_dim, lstm_hidden_dim, hidden_dim, input_dim)
        self.encoder = nn.Linear(self.feature_dim, hidden_dim)
        self.attn = SelfAttention(hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.cross_window_transformer = CrossWindowTransformer(embed_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            #nn.Sigmoid()
        )

    def generate(self, batch_size, device):
        z = torch.randn(batch_size, self.window_size, self.z_dim).to(device)
        return self.generator(z)


    def generate_n(self, batch_size, device, n_samples: int):
        z = torch.randn(n_samples, batch_size, self.window_size, self.z_dim, device=device)
        z = z.view(n_samples * batch_size, self.window_size, self.z_dim)
        x = self.generator(z)
        return x.view(n_samples, batch_size, -1)

    def discriminate(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.window_size, self.feature_dim)
        x_reshaped = x.reshape(-1, self.feature_dim)
        h = self.encoder(x_reshaped)
        h = F.leaky_relu(h, 0.2)
        h = h.view(batch_size, self.window_size, self.hidden_dim)
        h = self.attn(h)
        h = self.pos_encoding(h)
        h = self.cross_window_transformer(h)
        h = h.mean(dim=1)
        out = self.classifier(h)
        return out

class STGAD_NoSelfAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, z_dim=32, window_size=5, lstm_hidden_dim=64):
        super().__init__()
        self.name = 'STGAD_NoSelfAttn'
        self.lr = 0.0001
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.lstm_hidden_dim = lstm_hidden_dim
        assert input_dim % window_size == 0
        self.feature_dim = input_dim // window_size
        self.generator = LSTMGenerator(z_dim, lstm_hidden_dim, hidden_dim, input_dim)
        self.encoder = nn.Linear(self.feature_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.cross_window_transformer = CrossWindowTransformer(embed_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def generate(self, batch_size, device):
        z = torch.randn(batch_size, self.window_size, self.z_dim).to(device)
        return self.generator(z)


    def generate_n(self, batch_size, device, n_samples: int):
        z = torch.randn(n_samples, batch_size, self.window_size, self.z_dim, device=device)
        z = z.view(n_samples * batch_size, self.window_size, self.z_dim)
        x = self.generator(z)
        return x.view(n_samples, batch_size, -1)

    def discriminate(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.window_size, self.feature_dim)
        x_reshaped = x.reshape(-1, self.feature_dim)
        h = self.encoder(x_reshaped)
        h = F.leaky_relu(h, 0.2)
        h = h.view(batch_size, self.window_size, self.hidden_dim)
        h = self.pos_encoding(h)
        h = self.cross_window_transformer(h)
        h = h.mean(dim=1)
        out = self.classifier(h)
        return out
    
class STGAD_NoTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, z_dim=32, window_size=5, lstm_hidden_dim=64):
        super().__init__()
        self.name = 'STGAD_NoTransformer'
        self.lr = 0.0001
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.lstm_hidden_dim = lstm_hidden_dim
        assert input_dim % window_size == 0
        self.feature_dim = input_dim // window_size
        self.generator = LSTMGenerator(z_dim, lstm_hidden_dim, hidden_dim, input_dim)
        self.encoder = nn.Linear(self.feature_dim, hidden_dim)
        self.attn = SelfAttention(hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def generate(self, batch_size, device):
        z = torch.randn(batch_size, self.window_size, self.z_dim).to(device)
        return self.generator(z)


    def generate_n(self, batch_size, device, n_samples: int):
        z = torch.randn(n_samples, batch_size, self.window_size, self.z_dim, device=device)
        z = z.view(n_samples * batch_size, self.window_size, self.z_dim)
        x = self.generator(z)
        return x.view(n_samples, batch_size, -1)

    def discriminate(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.window_size, self.feature_dim)
        x_reshaped = x.reshape(-1, self.feature_dim)
        h = self.encoder(x_reshaped)
        h = F.leaky_relu(h, 0.2)
        h = h.view(batch_size, self.window_size, self.hidden_dim)
        h = self.attn(h)
        h = self.pos_encoding(h)
        h = h.mean(dim=1)
        out = self.classifier(h)
        return out