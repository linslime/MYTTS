import torch


def intersperse(lst, item):
	result = [item] * (len(lst) * 2 + 1)
	result[1::2] = lst
	return result


def get_padding(kernel_size, dilation=1):
	return int((kernel_size * dilation - dilation) / 2)


def sequence_mask(length, max_length=None):
	if max_length is None:
		max_length = length.max()
	x = torch.arange(max_length, dtype=length.dtype, device=length.device)
	return x.unsqueeze(0) < length.unsqueeze(1)


def init_weights(m, mean=0.0, std=0.01):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		m.weight.data.normal_(mean, std)

def convert_pad_shape(pad_shape):
	l = pad_shape[::-1]
	pad_shape = [item for sublist in l for item in sublist]
	return pad_shape

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
	n_channels_int = n_channels[0]
	in_act = input_a + input_b
	t_act = torch.tanh(in_act[:, :n_channels_int, :])
	s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
	acts = t_act * s_act
	return acts


def clip_grad_value_(parameters, clip_value, norm_type=2):
	if isinstance(parameters, torch.Tensor):
		parameters = [parameters]
	parameters = list(filter(lambda p: p.grad is not None, parameters))
	norm_type = float(norm_type)
	if clip_value is not None:
		clip_value = float(clip_value)
	
	total_norm = 0
	for p in parameters:
		param_norm = p.grad.data.norm(norm_type)
		total_norm += param_norm.item() ** norm_type
		if clip_value is not None:
			p.grad.data.clamp_(min=-clip_value, max=clip_value)
	total_norm = total_norm ** (1. / norm_type)
	return total_norm
