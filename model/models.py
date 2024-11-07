import numpy as np
from .Constants import *
from .FFTBlock import FFTBlock
from .attentions import Encoder
from model.modules import *
import commons


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


def sequence_mask(src_seq_length, max_length):
    x = torch.arange(max_length, dtype=src_seq_length.dtype, device=src_seq_length.device)
    return x.unsqueeze(0) > src_seq_length.unsqueeze(1)


class TextEncoder(nn.Module):
    """ Encoder """

    def __init__(self, n_vocab, max_seq_len, hidden_channels, encoder_layer, n_heads, conv_filter_size, conv_kernel_size, p_dropout):
        super(TextEncoder, self).__init__()
        self.n_vocab = n_vocab
        n_position = max_seq_len + 1
        d_k = d_v = hidden_channels // n_heads
        self.max_seq_len = max_seq_len
        self.d_model = hidden_channels

        self.src_word_emb = nn.Embedding(
            n_vocab, hidden_channels, padding_idx=PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, hidden_channels).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    hidden_channels, n_heads, d_k, d_v, conv_filter_size, conv_kernel_size, dropout=p_dropout
                )
                for _ in range(encoder_layer)
            ]
        )

    def forward(self, src_seq, src_seq_length, return_attns=False):
        mask = sequence_mask(src_seq_length, src_seq.size(1))
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        enc_output = self.src_word_emb(src_seq) + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, mask=mask, slf_attn_mask=slf_attn_mask)
        return enc_output


class StyleEncoder(nn.Module):
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)
        
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    
    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class PosteriorEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    
    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask



class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                                              gin_channels=gin_channels, mean_only=True))
            self.flows.append(Flip())
    
    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class StochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        
        self.log_flow = Log()
        self.flows = nn.ModuleList()
        self.flows.append(ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(Flip())
        
        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(Flip())
        
        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)
    
    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask
        
        if not reverse:
            flows = self.flows
            assert w is not None
            
            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q ** 2)) * x_mask, [1, 2]) - logdet_tot_q
            
            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2)) * x_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, max_seq_len, hidden_channels, decoder_layer, n_heads, conv_filter_size, conv_kernel_size, p_dropout):
        super(Decoder, self).__init__()

        n_position = max_seq_len + 1
        d_k = d_v = hidden_channels // n_heads

        self.max_seq_len = max_seq_len
        self.d_model = hidden_channels

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, hidden_channels).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    hidden_channels, n_heads, d_k, d_v, conv_filter_size, conv_kernel_size, dropout=p_dropout
                )
                for _ in range(decoder_layer)
            ]
        )

    def forward(self, enc_seq, mask):
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]
        max_len = min(max_len, self.max_seq_len)
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        dec_output = enc_seq[:, :max_len, :] + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        mask = mask[:, :max_len]
        slf_attn_mask = slf_attn_mask[:, :, :max_len]
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
        return dec_output, mask


class StyleAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, layer_number, hidden_channels, filter_size, kernel_size, dropout):
        super(StyleAdaptor, self).__init__()
        self.layer_number = layer_number
        # self.duration_predictor = VariancePredictor(hidden_channels, filter_size, kernel_size, dropout)
        self.encoder = nn.ModuleList(
            [
                StylePredictor(hidden_channels, filter_size, kernel_size, dropout)
                for _ in range(layer_number)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                StylePredictor(hidden_channels, filter_size, kernel_size, dropout)
                for _ in range(layer_number)
            ]
        )
        self.proj = nn.Conv1d(hidden_channels, hidden_channels * layer_number, 1)


    def forward(
        self,
        x,
        style_features,
        src_mask
    ):
        src_mask = torch.transpose(src_mask, 1, 2)
        style_features = self.proj(style_features)
        style_features = torch.transpose(style_features, 1, -1)
        out = x
        y = torch.split(style_features, x.size(2), 2)
        hidden_list = []
        for i, encoder in enumerate(self.encoder):
            out, hidden = encoder(out, y[i], src_mask)
            hidden_list.append(hidden)
        out = x
        for i, decoder in enumerate(self.decoder):
            out, _ = decoder(out, hidden_list[-i-1], src_mask)
        return out


class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    ):

        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)

        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        x = x.contiguous().transpose(1, 2)
        return x
