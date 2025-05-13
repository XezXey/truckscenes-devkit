import torch as th
import numpy as np
import torch.nn as nn

class MLP(th.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, condition_dim, model_channels, dropout, cfg):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert self.in_channels == self.out_channels
        self.model_channels = model_channels
        self.num_layers = num_layers
        self.condition_dim = condition_dim
        self.dropout = dropout
        self.cfg = cfg

        # Positional Embedding
        self.pos_encoder = PositionalEncoding(self.model_channels, self.dropout)
        # Timestep Embedding
        self.time_embed = TimestepEmbedder(latent_dim=self.model_channels, 
                                           sequence_pos_encoder=self.pos_encoder
                                           )

        self.cond_proj_layer = th.nn.Linear(self.condition_dim, self.model_channels)

        # Input Embedding (Processing the input e.g., project into the latent before pass through the mlp)
        self.input_process = th.nn.Linear(self.in_channels, self.model_channels)

        self.mlp_layers = []
        for i in range(self.num_layers):
            if i == 0:
                self.mlp_layers.append(th.nn.Linear(self.model_channels + self.model_channels + self.model_channels, self.model_channels))
                self.mlp_layers.append(th.nn.SiLU())
            else:
                self.mlp_layers.append(th.nn.Linear(self.model_channels, self.model_channels))
                self.mlp_layers.append(th.nn.SiLU())
        self.mlp_layers = th.nn.Sequential(*self.mlp_layers)

        self.output_process = th.nn.Linear(self.model_channels, self.out_channels)


    def forward(self, x, timesteps, **kwargs):
        """
        x: B x T x D, denoted x_t in the paper (input)
        timesteps: [batch_size, nframes] (input)
        """

        B, T, D = x.shape
        # Condition Embedding
        cond_emb = kwargs['cond']
        cond_emb_proj = self.cond_proj_layer(cond_emb)

        # Time Embedding
        t_emb = self.time_embed(timesteps)

        emb_cat = th.cat((cond_emb_proj.unsqueeze(dim=1), t_emb), dim=2)  # [bs, 1, d]
        emb_cat = emb_cat.squeeze(dim=1)  # [bs, d]


        # Flatten the input
        x = x.view(B, T * D)
        xp = self.input_process(x)   # [bs, T * D] -> [bs, d]; d=model_channels

        x_cat_emb = th.cat((emb_cat, xp), dim=1) # [bs, d_x] cat [bs, d_emb] -> [bs, d_x + d_emb]

        # Pass through the MLP layers
        output = self.mlp_layers(x_cat_emb)

        output = self.output_process(output)   # [bs, d_model] -> [bs, d_out]

        # Reshape the output back
        output = output.view(B, T, D)

        assert output.shape == (B, T, D), f"[#] Final output shape {output.shape} does not match input shape {(B, T, D)}."

        return {'output': output}

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        # print("timesteps : ", timesteps)
        self.sequence_pos_encoder.pe = self.sequence_pos_encoder.pe.to(timesteps.device)
        # print("pe shape : ", self.sequence_pos_encoder.pe.shape)
        # print("timesteps shape : ", self.sequence_pos_encoder.pe[timesteps].shape)
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = th.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = th.sin(position * div_term)
        pe[:, 0, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.permute(1, 0, 2)   # B x T x D -> T x B x D
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2)
        return self.dropout(x)