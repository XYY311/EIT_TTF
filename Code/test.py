"""
Stage3: FCN + decoder, FCN from the second stage but decoder from first stage
"""
import os
from torch.utils.data import Dataset
import numpy as np
import torch
from matplotlib import pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from dival.measure import PSNR, SSIM
# from dataset_paper import ys_test, xs_test
import pandas as pd
import numpy as np
import hdf5storage
import time
import scipy.io as sio
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda")


def min_max_normalize(matrix):
    min_value = np.min(matrix)
    max_value = np.max(matrix)
    normalized_matrix = (matrix - min_value) / (max_value - min_value)
    return normalized_matrix

class MyDataset(Dataset):
    def __init__(self,csv_path):

        # self.x_data = dd[0]
        # self.y_data = dd[1]
        # self.length = len(self.y_data)
        self.data_info = pd.read_csv(csv_path, encoding='gbk')
        # 文件第一列包含图像文件的名称
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # return self.x_data[index], self.y_data[index]
        data = self.image_arr[index]
        mat_data = hdf5storage.loadmat(data)

        U = mat_data['U']
        U = min_max_normalize(U)
        Sigma = mat_data['Sigma']
        rec_Sigma = mat_data['rec_Sigma']

        U = torch.Tensor(U)
        Sigma = torch.Tensor(Sigma)
        rec_Sigma = torch.Tensor(rec_Sigma)

        Sigma = np.reshape(Sigma, [128, 128])
        rec_Sigma = np.reshape(rec_Sigma, [128, 128])
        Node = mat_data['Node']
        Index = mat_data['Index']
        result = data.split("/")
        fileName = result[-1]

        # Sigma = min_max_normalize(Sigma.detach().numpy())
        # return U, Sigma,Node,Index,fileName
        # return Sigma,rec_Sigma,Node,Index,fileName
        return U, Sigma,rec_Sigma,Node,Index,fileName
        # return U, Sigma,rec_Sigma
    def __len__(self):
        # return self.length
        return self.data_len

dd_train = r'../data_csv/train_data_Sigma_reSigma_New.csv'
dd_val = r'../data_csv/test_data_Sigma_reSigma_New.csv'
# dd_train = r'../data_csv/train_data_all_Liver_ReSigma.csv'
# dd_val = r'../data_csv/test_data_all_Liver_ReSigma.csv'
train_dataset = MyDataset(dd_train)
val_dataset=MyDataset(dd_val)

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def Re_sigma(x, reco_eit):
    Re_up = torch.nn.L1Loss()(x, reco_eit)
    t = torch.zeros(x.shape).cpu()
    Re_down = torch.nn.L1Loss()(x, t)
    Re_sigma = Re_up / Re_down
    return Re_sigma

def DR(x, reco_eit):
    DR_up = torch.max(reco_eit) - torch.min(reco_eit)
    DR_down = torch.max(x) - torch.min(x)
    DR = DR_up / DR_down
    return DR

def samples(dataset,sample_size):
    """
    :return: sampler(index)
    """
    sampler = torch.utils.data.sampler.SubsetRandomSampler(
        np.random.choice(range(len(dataset)), sample_size))
    return sampler
batch_size =1
# device = torch.device('cuda')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dd_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
dd_val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False, drop_last=True)




# pymodel
unet = torch.load("mode/train_two_branch_210_20.pt", map_location=torch.device('cpu')) # change to your own vae model path
# unet = torch.load("mode_Liver/train_two_branch_121_20_0.9980011504820516.pt", map_location=torch.device('cpu')) # change to your own vae model path

unet.eval().to(device)

# vae_fcn_writer = SummaryWriter("vae_2/vae_fcn_")
savePath = 'H:\EIT all materials\PythonProject\CVAE/train_New_method/test_result/SAR-CGAN/'

mse_all =0; AE_all = 0; re_sigma_all = 0;  dr_all = 0;  ssim_all = 0;
mse = []; re = []; ssim = []; psnr = []
num = 0;
for idx, (U, Sigma,rec_Sigma,Node,Index,fileName) in enumerate(dd_val_loader):
    Sigma = Sigma.to(device)
    rec_Sigma = rec_Sigma.to(device)
    U = U.to(device)
    # U = U.unsqueeze(1)
    Sigma = Sigma.unsqueeze(1)
    rec_Sigma = rec_Sigma.unsqueeze(1)
    start_time = time.time()

    reco_eit1 = unet(U,rec_Sigma)
    end_time = time.time()
    # mse_man = np.mean([((xi.cpu() - gti.cpu()) ** 2).mean() for (xi, gti) in zip(reco_eit1, Sigma)])
    # mse_man = np.mean([(torch.nn.MSELoss(xi,gti)).mean() for (xi, gti) in zip(reco_eit1, Sigma)])
    # mse = torch.nn.MSELoss()(Sigma.data.squeeze(), reco_eit1.data.squeeze())
    # re_sigma = Re_sigma(Sigma.data.squeeze().cpu(), reco_eit1.data.squeeze().cpu())
    # AE = torch.nn.L1Loss()(Sigma.data.squeeze(), reco_eit1.data.squeeze())
    # dr = DR(Sigma.data.squeeze(), reco_eit1.data.squeeze())
    # ssim = SSIM(Sigma.data.cpu().numpy().squeeze(), reco_eit1.data.cpu().numpy().squeeze())
    # psnr = PSNR(Sigma.data.cpu().numpy().squeeze(), reco_eit1.data.cpu().numpy().squeeze())
    test_time = end_time - start_time

    psnr.append(PSNR(Sigma.data.cpu().numpy().squeeze(), reco_eit1.data.cpu().numpy().squeeze()))
    ssim.append(SSIM(Sigma.data.cpu().numpy().squeeze(), reco_eit1.data.cpu().numpy().squeeze()))
    mse.append(torch.nn.MSELoss()(Sigma.data.squeeze(), reco_eit1.data.squeeze()))
    re.append(Re_sigma(Sigma.data.squeeze().cpu(), reco_eit1.data.squeeze().cpu()))
    num+=1
    # AE_all = AE_all +AE
    # mse_all = mse_all+mse
    # ssim_all = ssim_all+ssim
    # dr_all = dr_all+dr
    # re_sigma_all=re_sigma_all+re_sigma
    #
    # for bs in range(batch_size):
    #     Sigma1 = Sigma[bs].view(1,-1).cpu().detach().numpy()
    #     U1 = U[bs].view(1,-1).cpu().detach().numpy()
    #     reco_eit11 = reco_eit1[bs].view(1,-1).cpu().detach().numpy()
    #     Node1 = Node[bs].view(2, -1).cpu().detach().numpy()
    #     Index1 = Index[bs].view(3,-1).cpu().detach().numpy()
    #     rec_Sigma1 = reco_eit11
    #     path = savePath+fileName[bs]
    #     sio.savemat(path,{'U':U1,'Sigma':Sigma1,'rec_Sigma':rec_Sigma1,'Node':Node1,'Index':Index1})
    #     del Sigma1, U1,reco_eit11,Node1,Index1,rec_Sigma1,path

    # Sigma = Sigma.view(1, -1).cpu().detach().numpy()
    # U = U.view(1, -1).cpu().detach().numpy()
    # reco_eit1 = reco_eit1.view(1, -1).cpu().detach().numpy()
    # Node = Node.view(2, -1).cpu().detach().numpy()
    # Index = Index.view(3, -1).cpu().detach().numpy()
    # rec_Sigma2 = reco_eit1
    # path = savePath + fileName[0]
    # sio.savemat(path, {'U': U, 'Sigma': Sigma, 'rec_Sigma': rec_Sigma2, 'Node': Node, 'Index': Index})


psnr = np.array(psnr)
ssim = np.array(ssim)
tensor_array_list = [tensor.cpu().numpy() for tensor in mse]
mse = np.array(tensor_array_list)
numpy_array_list = [tensor_re.numpy() for tensor_re in re]
re = np.array(numpy_array_list)


print("psnr_mean:%.4f,psnr_std:%.4f,ssim_mean:%.4f,ssim_std:%.4f,mse_mean:%.4f,mse_std:%.4f,re_mean:%.4f,re_std:%.4f"%(
    np.mean(psnr),np.std(psnr),np.mean(ssim),np.std(ssim),np.mean(mse),np.std(mse),np.mean(re),np.std(re)
))
# print("AE_all:%.4f,mse_all: %.4f,ssim_all: %.4f,dr_all: %.4f,re_sigma_all: %.4f" % (
#                 AE_all/num, mse_all/num, ssim_all/num, dr_all/num, re_sigma_all/num))

