import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from academicodec.models.ticodec.env import AttrDict, build_env
from academicodec.models.ticodec.meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from academicodec.models.encodec.msstftd import MultiScaleSTFTDiscriminator
from academicodec.models.ticodec.models import Generator
from academicodec.models.ticodec.models import MultiPeriodDiscriminator
from academicodec.models.ticodec.models import MultiScaleDiscriminator
from academicodec.models.ticodec.models import feature_loss
from academicodec.models.ticodec.models import generator_loss
from academicodec.models.ticodec.models import discriminator_loss
from academicodec.models.ticodec.models import Encoder
from academicodec.models.ticodec.models import Quantizer
from academicodec.utils import plot_spectrogram
from academicodec.utils import scan_checkpoint
from academicodec.utils import load_checkpoint
from academicodec.utils import save_checkpoint
from tqdm import tqdm
import logging
from datetime import datetime

torch.backends.cudnn.benchmark = True


def reconstruction_loss(x, G_x, device, eps=1e-7):
    L = 100 * F.mse_loss(x, G_x)  # wav L1 loss
    for i in range(6, 11):
        s = 2**i
        melspec = MelSpectrogram(
            sample_rate=24000,
            n_fft=s,
            hop_length=s // 4,
            n_mels=64,
            wkwargs={"device": device}).to(device)
        # 64, 16, 64
        # 128, 32, 128
        # 256, 64, 256
        # 512, 128, 512
        # 1024, 256, 1024
        S_x = melspec(x)
        S_G_x = melspec(G_x)
        loss = ((S_x - S_G_x).abs().mean() + (
            ((torch.log(S_x.abs() + eps) - torch.log(S_G_x.abs() + eps))**2
             ).mean(dim=-2)**0.5).mean()) / (i)
        L += loss
        #print('i ,loss ', i, loss)
    #assert 1==2
    return L


def setup_logger(
    log_filename: str,
    log_level: str = "info",
    use_console: bool = True,
) -> None:
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        formatter = f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] ({rank}/{world_size}) %(message)s"  # noqa
        log_filename = f"{log_filename}-{date_time}-{rank}"
    else:
        formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
        log_filename = f"{log_filename}-{date_time}"

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    level = logging.ERROR
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "critical":
        level = logging.CRITICAL

    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=level,
        filemode="w",
    )
    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger("").addHandler(console)


def train(rank, a, h):
    setup_logger(f"{a.checkpoint_path}/log/log-train")
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    encoder = Encoder(h).to(device)
    # gst = GST().to(device)
    # gst = Proposed(n_specs=128, token_num=10, E=128, n_layers=4).to(device)
    generator = Generator(h).to(device)
    quantizer = Quantizer(h).to(device)

    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    mstftd = MultiScaleSTFTDiscriminator(32).to(device)
    if rank == 0:
        logging.info(str(encoder))
        # logging.info(str(gst))
        logging.info(str(quantizer))
        logging.info(str(generator))
        os.makedirs(a.checkpoint_path, exist_ok=True)
        logging.info(f"checkpoints directory : {a.checkpoint_path}")

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        encoder.load_state_dict(state_dict_g['encoder'])
        # gst.load_state_dict(state_dict_g['gst'])
        quantizer.load_state_dict(state_dict_g['quantizer'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        mstftd.load_state_dict(state_dict_do['mstftd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(
            generator, device_ids=[rank]).to(device)
        encoder = DistributedDataParallel(encoder, device_ids=[rank]).to(device)
        # gst = DistributedDataParallel(gst, device_ids=[rank]).to(device)
        quantizer = DistributedDataParallel(
            quantizer, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)
        mstftd = DistributedDataParallel(mstftd, device_ids=[rank]).to(device)

    optim_g = torch.optim.Adam(
        itertools.chain(generator.parameters(),
                        encoder.parameters(), 
                        # gst.parameters(),
                        quantizer.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.Adam(
        itertools.chain(msd.parameters(), mpd.parameters(),
                        mstftd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2])
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(
        training_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        n_cache_reuse=0,
        shuffle=False if h.num_gpus > 1 else True,
        fmax_loss=h.fmax_for_loss,
        device=device,
        fine_tuning=a.fine_tuning,
        base_mels_path=a.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True)

    if rank == 0:
        validset = MelDataset(
            validation_filelist,
            h.segment_size,
            h.n_fft,
            h.num_mels,
            h.hop_size,
            h.win_size,
            h.sampling_rate,
            h.fmin,
            h.fmax,
            False,
            False,
            n_cache_reuse=0,
            fmax_loss=h.fmax_for_loss,
            device=device,
            fine_tuning=a.fine_tuning,
            base_mels_path=a.input_mels_dir)
        validation_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True)
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
    plot_gt_once = False
    generator.train()
    encoder.train()
    # gst.train()
    quantizer.train()
    mpd.train()
    msd.train()
    mstftd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            logging.info(f"Epoch: {epoch + 1}")
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel, y_global = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_global = torch.autograd.Variable(y_global.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)
            y_global = y_global.unsqueeze(1)

            c, global_features, global_features2 = encoder(y, y_global) # c: [batch_size, 512, T//320] global_features: [batch_size, 512]
            # print("c.shape: ", c.shape)
            # mid = mid.transpose(1, 2).unsqueeze(1)
            # global_style = gst(mid)
            # global_style = global_style.squeeze(1)
            global_features_compare_loss = 1-F.cosine_similarity(global_features, global_features2, dim=1).mean()
            # global_features_compare_loss = F.mse_loss(global_features, global_features2)
            q, loss_q, local_token, g, global_style_token = quantizer(c, global_features)

            # q, c, loss_q = residual_vq(y) # use residual vq modoule

            # if (0): # eval if is equal
            #     gen_q = quantizer.embed(torch.stack([code.reshape(h.batch_size, -1) for code in local_token], -1))
            #     gen_g = quantizer.embed_gst(torch.stack(global_style_token, -1).unsqueeze(1)).squeeze(-1)
            #     # print(gen_g)
            #     # print(g)
            #     # print(gen_q)
            #     # print(q)
            #     import numpy as np
            #     q_are_equal = np.array_equal(gen_q.detach().cpu().numpy(), q.detach().cpu().numpy())
            #     print(q_are_equal)
            #     g_are_equal = np.array_equal(gen_g.detach().cpu().numpy(), g.detach().cpu().numpy())
            #     print(g_are_equal)
            #     q_difference_sum = np.sum(np.abs(gen_q.detach().cpu().numpy() - q.detach().cpu().numpy()))
            #     print(q_difference_sum)
            #     g_difference_sum = np.sum(np.abs(gen_g.detach().cpu().numpy() - g.detach().cpu().numpy()))
            #     print(g_difference_sum)

            y_g_hat = generator(q, g)
            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                h.hop_size, h.win_size, h.fmin,
                h.fmax_for_loss)  # 1024, 80, 24000, 240,1024
            y_r_mel_1 = mel_spectrogram(
                y.squeeze(1), 512, h.num_mels, h.sampling_rate, 120, 512,
                h.fmin, h.fmax_for_loss)
            y_g_mel_1 = mel_spectrogram(
                y_g_hat.squeeze(1), 512, h.num_mels, h.sampling_rate, 120, 512,
                h.fmin, h.fmax_for_loss)
            y_r_mel_2 = mel_spectrogram(
                y.squeeze(1), 256, h.num_mels, h.sampling_rate, 60, 256, h.fmin,
                h.fmax_for_loss)
            y_g_mel_2 = mel_spectrogram(
                y_g_hat.squeeze(1), 256, h.num_mels, h.sampling_rate, 60, 256,
                h.fmin, h.fmax_for_loss)
            y_r_mel_3 = mel_spectrogram(
                y.squeeze(1), 128, h.num_mels, h.sampling_rate, 30, 128, h.fmin,
                h.fmax_for_loss)
            y_g_mel_3 = mel_spectrogram(
                y_g_hat.squeeze(1), 128, h.num_mels, h.sampling_rate, 30, 128,
                h.fmin, h.fmax_for_loss)
            # print("x.shape: ", x.shape)
            # print("y.shape: ", y.shape)
            # print("y_g_hat.shape: ", y_g_hat.shape)
            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                y_ds_hat_r, y_ds_hat_g)

            y_disc_r, fmap_r = mstftd(y)
            y_disc_gen, fmap_gen = mstftd(y_g_hat.detach())
            loss_disc_stft, losses_disc_stft_r, losses_disc_stft_g = discriminator_loss(
                y_disc_r, y_disc_gen)
            loss_disc_all = loss_disc_s + loss_disc_f + loss_disc_stft

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel1 = F.l1_loss(y_r_mel_1, y_g_mel_1)
            loss_mel2 = F.l1_loss(y_r_mel_2, y_g_mel_2)
            loss_mel3 = F.l1_loss(y_r_mel_3, y_g_mel_3)
            #print('loss_mel1, loss_mel2 ', loss_mel1, loss_mel2)
            loss_mel = F.l1_loss(y_mel,
                                 y_g_hat_mel) * 45 + loss_mel1 + loss_mel2
            # print('loss_mel ', loss_mel)
            # assert 1==2
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            y_stftd_hat_r, fmap_stftd_r = mstftd(y)
            y_stftd_hat_g, fmap_stftd_g = mstftd(y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_fm_stft = feature_loss(fmap_stftd_r, fmap_stftd_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_stft, losses_gen_stft = generator_loss(y_stftd_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_gen_stft + loss_fm_s + loss_fm_f + loss_fm_stft + loss_mel + loss_q * 10 + global_features_compare_loss * 5
            loss_gen_all.backward()
            optim_g.step()
            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                    logging.info(
                        f"Steps : {steps:d}, Gen Loss Total : {loss_gen_all:4.3f}, Loss Q : {loss_q:4.3f}, Mel-Spec. Error : {mel_error:4.3f}, s/b : {time.time() - start_b:4.3f}")
                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path,
                                                           steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'generator': (generator.module if h.num_gpus > 1
                                          else generator).state_dict(),
                            'encoder': (encoder.module if h.num_gpus > 1 else
                                        encoder).state_dict(),
                            # 'gst': (gst.module if h.num_gpus > 1 else gst).state_dict(),
                            'quantizer': (quantizer.module if h.num_gpus > 1
                                          else quantizer).state_dict()
                        },
                        num_ckpt_keep=a.num_ckpt_keep)
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path,
                                                            steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'mpd': (mpd.module
                                    if h.num_gpus > 1 else mpd).state_dict(),
                            'msd': (msd.module
                                    if h.num_gpus > 1 else msd).state_dict(),
                            'mstftd': (mstftd.module
                                       if h.num_gpus > 1 else mstftd).state_dict(),
                            'optim_g':
                            optim_g.state_dict(),
                            'optim_d':
                            optim_d.state_dict(),
                            'steps':
                            steps,
                            'epoch':
                            epoch
                        },
                        num_ckpt_keep=a.num_ckpt_keep)
                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/disc_loss_total", loss_disc_all, steps)
                    sw.add_scalar("training/loss_mel", loss_mel, steps)
                    sw.add_scalar("training/loss_quantizer", loss_q, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    sw.add_scalar("training/compare_loss", global_features_compare_loss, steps)

                # Validation
                if steps % a.validation_interval == 0 and steps != 0:
                    generator.eval()
                    encoder.eval()
                    # gst.eval()
                    quantizer.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in tqdm(enumerate(validation_loader)):
                            x, y, _, y_mel = batch
                            c, global_features = encoder(y.to(device).unsqueeze(1))
                            # mid = mid.transpose(1, 2).unsqueeze(1)
                            # global_style = gst(mid)
                            q, loss_q, local_token, g, global_style_token = quantizer(c, global_features)
                            y_g_hat = generator(q, g)
                            y_mel = torch.autograd.Variable(y_mel.to(device))
                            y_g_hat_mel = mel_spectrogram(
                                y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                h.sampling_rate, h.hop_size, h.win_size, h.fmin,
                                h.fmax_for_loss)
                            i_size = min(y_mel.size(2), y_g_hat_mel.size(2))
                            val_err_tot += F.l1_loss(
                                y_mel[:, :, :i_size],
                                y_g_hat_mel[:, :, :i_size]).item()

                            # if j <= 8:
                            if j%10 == 1 or j%10 == 2:
                                # if steps == 0:
                                if not plot_gt_once:
                                    sw.add_audio('gt/y_{}'.format(j), y[0],
                                                 steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j),
                                                  plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j),
                                             y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(
                                    y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                    h.sampling_rate, h.hop_size, h.win_size,
                                    h.fmin, h.fmax)
                                sw.add_figure(
                                    'generated/y_hat_spec_{}'.format(j),
                                    plot_spectrogram(
                                        y_hat_spec.squeeze(0).cpu().numpy()),
                                    steps)

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err,
                                      steps)
                        if not plot_gt_once:
                            plot_gt_once = True

                    generator.train()
                    encoder.train()
                    # gst.train()
                    quantizer.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            logging.info(f"Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n")


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    # parser.add_argument('--group_name', default=None)
    # parser.add_argument('--input_wavs_dir', default='../datasets/audios')
    parser.add_argument('--input_mels_dir', default=None)
    parser.add_argument('--input_training_file', required=True)
    parser.add_argument('--input_validation_file', required=True)
    parser.add_argument('--checkpoint_path', default='checkpoints')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=2000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument('--num_ckpt_keep', default=100, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h, ))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
