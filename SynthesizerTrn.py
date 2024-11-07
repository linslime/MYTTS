from torch import nn
import torch
import math
import monotonic_align
from model import *


def get_attn(z_p, logs_p, m_p, x_mask, y_mask):
	with torch.no_grad():
		# negative cross-entropy
		s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
		neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_s]
		neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2),
		                         s_p_sq_r)  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
		neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
		neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
		neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
		
		attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
		attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
	return attn


class SynthesizerTrn(nn.Module):
	"""
	Synthesizer for Training
	"""
	def __init__(self, n_vocab,
	             max_seq_len,
	             layer_number,
	             spec_channels,
	             inter_channels,
	             hidden_channels,
	             filter_channels,
	             encoder_layer,
	             decoder_layer,
	             n_heads,
	             conv_filter_size,
	             conv_kernel_size,
	             kernel_size,
	             n_layers,
	             p_dropout):
		super().__init__()
		self.text_encoder = TextEncoder(n_vocab, max_seq_len, hidden_channels, encoder_layer, n_heads, conv_filter_size, conv_kernel_size, p_dropout)
		self.style_encoder = StyleEncoder(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
		self.dp = StochasticDurationPredictor(hidden_channels, filter_channels, kernel_size, p_dropout)
		self.style_adaptor = StyleAdaptor(layer_number, hidden_channels, 256, kernel_size, p_dropout)
		self.decoder = Decoder(max_seq_len, hidden_channels, decoder_layer, n_heads, conv_filter_size, conv_kernel_size, p_dropout)
		self.mel_linear = nn.Linear(192, 80)
		self.postnet = PostNet()
		self.style_extractor = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, 0)
		self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, 0)
	
	def forward(self, x, x_lengths, y, y_lengths, sid=None):
		output = self.text_encoder(x, x_lengths)
		x, m_p, logs_p, x_mask = self.style_encoder(x, x_lengths)
		z, m_q, logs_q, y_mask = self.style_extractor(y, y_lengths, g=None)
		z_p = self.flow(z, y_mask, g=None)
		attn = get_attn(z_p, logs_p, m_p, x_mask, y_mask)
		
		l_length = self.dp(x, x_mask, attn.sum(2), g=None) / torch.sum(x_mask)
		
		m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
		logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
		
		output = self.style_adaptor(output, x, x_mask)
		output = torch.matmul(attn.squeeze(1), output)
		x_mask = torch.transpose(x_mask, 1, 2).repeat(1, 1, output.size(2))
		x_mask = (1 - torch.matmul(attn.squeeze(1), x_mask)[:, :, 0]).bool()
		
		output, mel_masks = self.decoder(output, x_mask)
		output = self.mel_linear(output)
		output = self.postnet(output) + output
		output = torch.transpose(output, 1, 2)
		return output, l_length, attn, m_p, logs_p, m_q, logs_q, z_p, z, y_mask
	