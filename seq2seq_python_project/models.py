import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, hidden = self.gru(x)
        return hidden

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=2):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        out, _ = self.gru(x, hidden)
        predict = self.fc(out)
        return predict

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, future_seq_len, num_layers=2):
        super(Seq2Seq, self).__init__()
        self.future_seq_len = future_seq_len
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.decoder = Decoder(hidden_dim, output_dim, num_layers)

    def forward(self, x):
        hidden = self.encoder(x)
        context_vector = hidden[-1].unsqueeze(1)
        decoder_input = context_vector.repeat(1, self.future_seq_len, 1)
        predictions = self.decoder(decoder_input, hidden)
        return predictions

class BiImputationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hole_len, num_layers=2):
        super().__init__()
        self.hole_len = hole_len
        self.num_layers = num_layers
        
        self.enc_left = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.enc_right = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        self.dec_gru = nn.GRU(hidden_dim * 4, hidden_dim * 4, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 4, output_dim)

    def forward(self, x):
        mid = x.size(1) // 2
        part_left = x[:, :mid, :]
        part_right = x[:, mid:, :]
        
        _, h_left = self.enc_left(part_left)
        _, h_right = self.enc_right(part_right)
        
        hl = torch.cat([h_left[-2], h_left[-1]], dim=-1)
        hr = torch.cat([h_right[-2], h_right[-1]], dim=-1)
        
        context = torch.cat([hl, hr], dim=-1)
        
        decoder_in = context.unsqueeze(1).repeat(1, self.hole_len, 1)
        h0 = context.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        out, _ = self.dec_gru(decoder_in, h0)
        return self.fc(out)