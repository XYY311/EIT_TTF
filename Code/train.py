# -*- coding: utf-8 -*-
"""
Stage1: VAE:encoder+decoder. encoder for the second stage and decoder for the three stage
"""

import os

import torch.nn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from dival.measure import PSNR, SSIM
from matplotlib import pyplot as plt
import torch.optim as optim
import tqdm
from vae import *
# from dataset_paper import xs_train, xs_val
from torch.utils.tensorboard import SummaryWriter
import imlib as im
from dataset_new import *
from torch.optim.lr_scheduler import StepLR
from unet import *
from Unet_two_branch import UNet_two_branch

# device
device = torch.device("cuda")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs=301
learning_rate = 0.0003
batch_size = 20
xs_train = r'../data_csv/train_data_Sigma_reSigma_New.csv'
xs_val = r'../data_csv/test_data_Sigma_reSigma_New.csv'
# xs_train = r'../data_csv/train_data_all_Liver_ReSigma.csv'
# xs_val = r'../data_csv/test_data_all_Liver_ReSigma.csv'
loss_function = torch.nn.MSELoss()


x_train_loader = torch.utils.data.DataLoader(dataset(xs_train), batch_size, shuffle=True,drop_last=True)
x_val_loader = torch.utils.data.DataLoader(dataset(xs_val), batch_size, shuffle=False,drop_last=True)

unet_writer = SummaryWriter("path_{}_{}".format(epochs,  batch_size)) # change to your own path

unet = UNet_two_branch().to(device)
optimizer = optim.Adam(unet.parameters(), lr=learning_rate)
# optimizer = optim.SGD(vae.parameters(), lr=1e-4)
# optimizer = optim.Adam(vae.parameters(), lr=0.0005)
# 定义学习率调度器
max_ssim = 0
scheduler = StepLR(optimizer, step_size=10, gamma=0.00001)
for epoch in tqdm.trange(epochs, desc='Epoch Loop'):
    with tqdm.tqdm(x_train_loader, total=x_train_loader.batch_sampler.__len__()) as t:
        unet.train(mode=True)
        for idx, (U,Sigma,rec_Sigma) in enumerate(x_train_loader):
            U, Sigma,rec_Sigma = U.to(device),Sigma.to(device),rec_Sigma.to(device)
            U = U.unsqueeze(1)
            Sigma = Sigma.unsqueeze(1)
            rec_Sigma = rec_Sigma.unsqueeze(1)
            recon_images = unet(U,rec_Sigma)
            loss = loss_function(Sigma,recon_images)
            # loss, mse, kld = loss_fn(recon_images, Sigma, mu, logvar, batch_size)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            unet_writer.add_scalar("train_lr", optimizer.param_groups[0]["lr"], epoch)

            t.set_description(f'Epoch {epoch}')
            t.set_postfix(ordered_dict={'Loss': loss.item()})
            t.update(1)


        unet_writer.add_scalar("train_loss", loss, epoch)

        
        with tqdm.tqdm(x_val_loader, total=x_val_loader.batch_sampler.__len__()) as ti:
            unet.eval()
            ssim_all = 0
            num = 0
            ssim_value = 0
            with torch.no_grad():
                for idx, (U,Sigma,rec_Sigma) in enumerate(x_val_loader):
                    U,Sigma,rec_Sigma = U.to(device),Sigma.to(device),rec_Sigma.to(device)
                    U = U.unsqueeze(1)
                    Sigma = Sigma.unsqueeze(1)
                    rec_Sigma = rec_Sigma.unsqueeze(1)

                    val_recon_images = unet(U,rec_Sigma)
                    num = num+1
                    # val_loss, val_mse, val_kld = loss_fn(val_recon_images, Sigma, val_mu, val_logvar, batch_size)
                    val_loss = loss_function(Sigma,val_recon_images)

                    psnr_mean = np.mean([PSNR(xi.cpu(), gti.cpu()) for (xi, gti) in zip(val_recon_images, Sigma)])
                    ssim_mean = np.mean(
                        [SSIM(xi[0].cpu().numpy(), gti[0].cpu().numpy()) for (xi, gti) in zip(val_recon_images, Sigma)])
                    mse_mean = np.mean(
                        [((xi.cpu() - gti.cpu()) ** 2).mean() for (xi, gti) in zip(val_recon_images, Sigma)])
                    # metrics = {'psnr': psnr_mean, 'ssim': ssim_mean, 'mse': mse_mean}

                    unet_writer.add_scalar("psnr_mean", psnr_mean, epoch)
                    unet_writer.add_scalar("ssim_mean", ssim_mean, epoch)
                    unet_writer.add_scalar("mse_mean", mse_mean, epoch)
                    # unet_writer.add_scalar("val_mse", val_mse, epoch)
                    ti.set_description(f'Epoch {epoch}')
                    ti.set_postfix(ordered_dict={'psnr_mean': psnr_mean.item(),
                                                'ssim_mean': ssim_mean.item(),
                                                'mse_mean': mse_mean.item()})
                    ti.update(1)
                    ssim_all = ssim_all+ssim_mean



            ssim_value = ssim_all/num

        # if (ssim_value>= max_ssim):
        #     torch.save(unet, "mode/train_two_branch_{}_{}_{}.pt".format(epoch, batch_size,ssim_value))  # change to your own path
        #     max_ssim = ssim_value

    #
    if epoch % 30 == 0:
        torch.save(unet, "mode/train_two_branch_{}_{}.pt".format(epoch, batch_size))  # change to your own path


unet_writer.close()












