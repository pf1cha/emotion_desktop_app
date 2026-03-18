import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, pretrained_embeddings, hidden_dim, num_layers, num_classes, dropout_rate=0.5):
        super(BiLSTMClassifier, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        embed_dim = pretrained_embeddings.size(1)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout_rate)

        self.attention = nn.Linear(hidden_dim * 2, 1)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)

        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)

        context_vector = torch.sum(attn_weights * lstm_out, dim=1)

        last_hidden = self.dropout(context_vector)
        output = self.fc(last_hidden)
        return output
