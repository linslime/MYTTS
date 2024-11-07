import utils
from data_utils import *
from torch.utils.data import DataLoader
from SynthesizerTrn import SynthesizerTrn
from text.symbols import symbols
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from mel_processing import spec_to_mel_torch
from torch.nn import functional as F


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
	"""
	z_p, logs_q: [b, h, t_t]
	m_p, logs_p: [b, h, t_t]
	"""
	z_p = z_p.float()
	logs_q = logs_q.float()
	m_p = m_p.float()
	logs_p = logs_p.float()
	z_mask = z_mask.float()
	
	kl = logs_p - logs_q - 0.5
	kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2. * logs_p)
	kl = torch.sum(kl * z_mask)
	l = kl / torch.sum(z_mask)
	return l


def main():
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '6667'
	n_gpus = torch.cuda.device_count()
	hps = utils.get_hparams()
	mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
	dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
	torch.manual_seed(hps.train.seed)
	torch.cuda.set_device(rank)
	
	train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
	train_sampler = DistributedBucketSampler(
		train_dataset,
		hps.train.batch_size,
		[32, 300, 400, 500, 600, 700, 800, 900, 1000],
		num_replicas=n_gpus,
		rank=rank,
		shuffle=True)
	collate_fn = TextAudioCollate()
	train_loader = DataLoader(dataset=train_dataset, num_workers=8, pin_memory=True, collate_fn=collate_fn,
	                          batch_sampler=train_sampler)
	
	params = (len(symbols),
	          hps.max_seq_len,
	          hps.model.stage_layer,
	          hps.data.filter_length // 2 + 1,
	          hps.model.inter_channels,
	          hps.model.hidden_channels,
	          hps.model.filter_channels,
	          hps.model.encoder_layer,
	          hps.model.decoder_layer,
	          hps.model.n_heads,
	          hps.model.conv_filter_size,
	          hps.model.conv_kernel_size,
	          hps.model.kernel_size,
	          hps.model.n_layers,
	          hps.model.p_dropout)
	model = SynthesizerTrn(*params).cuda(rank)
	
	optim = torch.optim.AdamW(
		model.parameters(),
		hps.train.learning_rate,
		betas=hps.train.betas,
		eps=hps.train.eps)
	model = DDP(model, device_ids=[rank])
	epoch_str = 1
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
	scaler = GradScaler(enabled=hps.train.fp16_run)
	for epoch in range(epoch_str, hps.train.epochs + 1):
		train_loader.batch_sampler.set_epoch(epoch)
		model.train()
		for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
			x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
			spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
			y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
			with autocast(enabled=hps.train.fp16_run):
				output, l_length, attn, m_p, logs_p, m_q, logs_q, z_p, z, z_mask = model(x, x_lengths, spec, spec_lengths)
				mel = spec_to_mel_torch(
					spec,
					hps.data.filter_length,
					hps.data.n_mel_channels,
					hps.data.sampling_rate,
					hps.data.mel_fmin,
					hps.data.mel_fmax)
				with autocast(enabled=False):
					loss_dur = torch.sum(l_length.float())
					loss_mel = F.l1_loss(mel, output) * hps.train.c_mel
					loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
					loss_gen_all = loss_mel + loss_dur + loss_kl
			optim.zero_grad()
			scaler.scale(loss_gen_all).backward()
			scaler.unscale_(optim)
			scaler.step(optim)
			scaler.update()
		if epoch % 5 == 0:
			torch.save(model.state_dict(), './trained_model/' + str(epoch) + '.pt')
		scheduler.step()


if '__main__' == __name__:
	main()
