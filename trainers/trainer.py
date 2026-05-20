
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import torch
from torch import nn
from torch import distributions
if 'ilee141' not in os.getcwd():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import cuda, grid2gif, KLdivergence
from dataset import get_dataset
from models.model import get_model

def get_trainer(config):
    if config['type'] == 'beta':
        from trainers.beta_trainer import BetaTrainer
        return BetaTrainer(config)
    elif config['type'] == 'betatc':
        from trainers.betatc_trainer import BetatcTrainer
        return BetatcTrainer(config)
    elif config['type'] == 'factor':
        from trainers.factor_trainer import FactorTrainer
        return FactorTrainer(config)
    elif config['type'] == 'info':
        from trainers.info_trainer import InfoTrainer
        return InfoTrainer(config)
    else:
        return Trainer(config)

class Trainer(object):
    """docstring for Trainer."""

    def __init__(self, config):
        super(Trainer, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        # self.device ='cuda:1'

        # model
        self.modef = config['model']
        self.model = get_model(config)
        self.input_dims = config['input_dims']
        self.z_dims = config['z_dims']
        self.prior = distributions.MultivariateNormal(torch.zeros(self.z_dims), torch.eye(self.z_dims))

        # train
        self.max_iter = config['max_iter']
        self.global_iter = 1
        self.mseWeight = config['mse_weight']
        self.lr = config['lr']
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.optim = optim.Adam(
          self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )
        self.implicit = 'implicit' in config and config['implicit']
        if self.implicit:
            self.train_inst = self.implicit_inst

        # saving
        self.ckpt_dir = config['ckpt_dir']
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = config['ckpt_name']
        self.save_output = config['save_output']
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        # saving
        if config['cont'] and self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.meta = defaultdict(list)

        self.gather_step = config['gather_step']
        self.display_step = config['display_step']
        self.save_step = config['save_step']

        # data
        self.dset_dir = config['dset_dir']
        self.dataset = config['dataset']
        self.data_type = config['data_type']
        if self.data_type == 'linear':
            self.draw_reconstruction = self.linear_reconstruction
            self.draw_generated = self.linear_generated
            self.visualize_traverse = self.linear_traverse
            self.traversal = self.linear_traversal
        self.batch_size = config['batch_size']
        self.img_size = 32 if 'image_size' not in config else config['image_size']
        self.data_loader = get_dataset(config)
        self.val_loader = get_dataset(config, train=False)
        # for k in self.data_loader:
        #     print(k)
    def train(self):
        self.model.train()
        meta = defaultdict(lambda: 0)
        best = 10000000
        loop = tqdm(total=self.max_iter, position=0)
        while self.global_iter < self.max_iter+1:
            for x in self.data_loader:

                x, x_recon, loss = self.train_inst(x)
                for key in loss:
                    meta[key] += loss[key]
                if self.global_iter%self.gather_step == 0:
                    for key in meta:
                        self.meta[key].append(meta[key]/self.gather_step)
                        meta[key] = 0

                if self.global_iter%self.display_step == 0:
                    loop.write(self.loop_update)

                    self.draw_reconstruction()
                    self.draw_generated()
                    self.draw_qz()
                    self.draw_loss()
                    # self.traversal()
                    # self.visualize_traverse()
                    # self.computeGenMetrics()

                if self.global_iter%self.save_step == 0:
                    self.save_checkpoint(str(self.global_iter))
                    loop.write('Saved checkpoint(iter:{})'.format(self.global_iter))
                    if self.meta['loss'][-1] < best:
                        self.save_checkpoint('best')
                        best = self.meta['loss'][-1]

                loop.update(1)
                self.global_iter += 1
                if self.global_iter > self.max_iter:
                    break
        loop.write("[Training Finished]")
        loop.close()

    def train_inst(self, x):
        x = x.to(self.device)
        x_recon, mu, logvar, z = self.model(x)
        recon_loss = self.reconstruction_loss(x, x_recon)
        kld = self.kl_divergence(mu, logvar)
        loss = self.loss(recon_loss, kld)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        meta = {}
        meta['mus'] = mu.mean(0).data
        meta['vars'] = logvar.exp().mean(0).data
        meta['recon_loss'] = recon_loss.item()
        meta['kld'] = kld.item()
        meta['loss'] = loss.item()

        if self.global_iter%self.display_step == 0:
            self.loop_updater(recon_loss, kld, loss)
            # self.add_logvar(logvar)

        return x, torch.sigmoid(x_recon), meta

    def implicit_inst(self, x):
        x = x.to(self.device)
        x_recon, mu, logvar, z, g_1, g_2 = self.model(x)
        # likelihood loss
        recon_loss = self.reconstruction_loss(x, x_recon)
		# DKL loss
        ELS = -1*torch.sum(g_2)
        k = torch.mul(g_1, g_1).sum(dim=1)
        DKL = -0.5*torch.sum(logvar.sum(dim=1) - k)
        kld = ELS+DKL

        loss = self.loss(recon_loss, kld)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        meta = {}
        meta['mus'] = mu.mean(0).data
        meta['vars'] = logvar.exp().mean(0).data
        meta['recon_loss'] = recon_loss.item()
        meta['ELS'] = ELS.item()
        meta['kld'] = kld.item()
        meta['loss'] = loss.item()

        if self.global_iter%self.display_step == 0:
            self.loop_updater(recon_loss, kld, loss)
            # self.add_logvar(logvar)

        return x, torch.sigmoid(x_recon), meta

    # loss functions

    def loss(self, recon_loss, kld):
        return recon_loss + kld

    def reconstruction_loss(self, x, x_recon):
        loss = self.mseWeight*F.mse_loss(
          x_recon, x, reduction='sum'
        )
        return loss

    def kl_divergence(self, mu, logvar):
        # return -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
        return -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def loop_updater(self, recon_loss, kld, loss):
        self.loop_update = '[{}] recon_loss:{:.3f} kld:{:.3f} loss:{:.3f}'.format(
          self.global_iter, recon_loss.item(), kld.item(), loss.item()
        )

    def add_logvar(self, logvar):
        var = logvar.exp().mean(0).data
        var_str = ' '
        for j, var_j in enumerate(var):
            var_str += 'var{}:{:.4f} '.format(j+1, var_j)
        self.loop_update = self.loop_update + var_str

    # visualization methods

    def saveZspace(self):
        ''' saving trainigna and validation data for SVN classifiaction'''
        Dl = len(self.data_loader.dataset)
        trainMU = np.zeros([Dl, self.z_dims])
        count = 0
        for x in self.data_loader:
            qzs = self.model.encoder(x.to(self.device))
            mus = qzs[:, :self.z_dims]
            trainMU[count*self.batch_size:(count+1)*self.batch_size, :] = mus.cpu().detach().numpy()
            count += 1

        Dl = len(self.val_loader.dataset)
        # valMU = np.zeros([Dl, self.z_dims])
        valImg = np.zeros([Dl, self.img_size, self.img_size])
        count = 0
        for x in self.val_loader:
            qzs = self.model.encoder(x.to(self.device))
            mus = qzs[:, :self.z_dims]
            temp = x.cpu().detach().numpy()
            temp = temp.reshape(self.batch_size, self.img_size, self.img_size)
            # valImg[count*self.batch_size:(count+1)*self.batch_size, ...] = temp
            valMU[count*self.batch_size:(count+1)*self.batch_size, :] = mus.cpu().detach().numpy()
            count += 1
        output_dir = osp.join(self.output_dir, str(self.global_iter))
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir + 'trainLatent.npy', trainMU)
        np.save(output_dir + 'testLatent.npy', valMU)
        # np.save(output_dir + 'testImg.npy', valImg)

    def draw_reconstruction(self, x=None, x_recon=None,
      ori_name='original.png', rec_name='reconstructed.png'):
        output_dir = osp.join(self.output_dir, str(self.global_iter))
        os.makedirs(output_dir, exist_ok=True)
        if x is None:
            if self.data_type == 'linear':
                x = torch.cat( [x for x in self.val_loader], dim=0 )
            else:
                for i, y in enumerate(self.val_loader):
                    x = y
                    break
        if x_recon is None:
            x_recon = self.model(x.to(self.device))[0].detach()
        xgrid = make_grid(x)
        save_image(tensor=xgrid.cpu(), nrow=self.z_dims*x.size(0), pad_value=1,
          filename=osp.join(output_dir, ori_name)
        )
        # genpath = osp.join(output_dir, 'original')
        # os.makedirs(genpath, exist_ok=True)
        # for i in range(self.batch_size):
        #     temp = x[i, ...]
        #     save_image(tensor=temp.cpu(), nrow=self.z_dims*x.size(0), pad_value=1,
        #     filename=osp.join(genpath, 'original' + str(i) + '.png'))
        recon_grid = make_grid(x_recon)
        save_image(tensor=recon_grid.cpu(), nrow=self.z_dims*x_recon.size(0), pad_value=1,
          filename=osp.join(output_dir, rec_name)
        )

    def highLow_draw_reconstruction(self, x_low, x_high, low_name, high_name):
        output_dir = osp.join(self.output_dir, str(self.global_iter))
        os.makedirs(output_dir, exist_ok=True)
        # xgrid = make_grid(x_low)
        save_image(tensor=x_low.cpu(), nrow=6, padding=0,
          filename=osp.join(output_dir, low_name)
        )
        # recon_grid = make_grid(x_high)
        save_image(tensor=x_high.cpu(), nrow=6, padding=0,
          filename=osp.join(output_dir, high_name)
        )

    def linear_reconstruction(self, x=None, x_recon=None):
        output_dir = osp.join(self.output_dir, str(self.global_iter))
        os.makedirs(output_dir, exist_ok=True)
        if x is None:
            x = torch.cat( [x for x in self.val_loader], dim=0 )
        if x_recon is None:
            x_recon = self.model(x.to(self.device))[0].detach()
        x = x.cpu().data.numpy()
        plt.scatter(x[:,0], x[:,1])
        plt.savefig(osp.join(output_dir, 'original.png'))
        plt.clf()
        x_recon = x_recon.cpu().data.numpy()
        plt.scatter(x_recon[:,0], x_recon[:,1])
        plt.savefig(osp.join(output_dir, 'reconstructed.png'))
        plt.close()

    def generateSamples(self, numGen):
        # self.generate_normal_samples(numGen)
        # self.generate_lowpos_samples(numGen)
        self.generate_highpos_samples(numGen)
        # self.generate_original()

    def probMUltiGaussian(self, sample, mu, sig):
        sig_mat = np.diag(sig)
        L = len(sample)
        r = (sample - mu).reshape(L, 1)
        temp = -0.5*np.matmul(np.matmul(r.T, sig_mat), r) - 0.5* np.log(2*np.pi * np.linalg.det(sig_mat))
        return temp
    
    def qz_prob(self, sample, muAll, sigAll):
        # N = len(self.val_loader.dataset)
        N = 5000
        add = 0
        for i in range(N):
            add += np.exp(self.probMUltiGaussian(sample, muAll[i, :], sigAll[i, :]))
        out = add/N
        out = np.log(out)
        return out

    def analyze_zspace(self, x, covMat):
        # compute the gradients of the decoder
        # get the output corresponding to the input
        # x_hat = self.model(x.to(self.device))[0]
        # we also want the z value
        qzs = self.model.encoder(x.to(self.device))
        mu = qzs[:, :self.z_dims]
        logvar = qzs[:, self.z_dims:]
        z = self.model.reparametrize(mu, logvar)
        x_hat = self.model.decoder(z)
        err = torch.mean((x.to(self.device) - x_hat)**2)
        # Now you want the jacobian of X_hat wrt z
        
        if self.implicit:
            z = self.model.g(z)[0]
            zz = self.model.g_inv(z)
            x_hat = self.model.decoder(zz)

        x_hat_flat = x_hat.view(-1, 28*28)
        jac = torch.zeros(28*28, 2).to(self.device)
        for i in range(28*28):
            jac[i, :] = torch.autograd.grad(x_hat_flat[0,i], z, retain_graph=True)[0]
            # jac[i, 1] = torch.autograd.grad(x_hat_flat[0,i], z)[1]
        jac_det = torch.det(torch.matmul(jac.t(), jac))**0.5
        # now compute the prob of the Z in standard normal
        pz = self.compute_normal_prob(z)
        px_onmanifold = pz/jac_det
        # compute the off manifold probability which is just the reconstruction
        # px_offmanifold = self.compute_MVnorm_prob(x.view(-1, 28*28), x_hat_flat, covMat)
        px_offmanifold = 1/err
        px = px_onmanifold*px_offmanifold
        # print(px_offmanifold, px_onmanifold)
        return [px, px_offmanifold, px_onmanifold]

    def compute_MVnorm_prob(self, X, mu, covMat):
        d = 28*28
        covMat = covMat.to(self.device)
        res = (X.to(self.device) - mu)
        num = torch.matmul(torch.matmul(res, torch.inverse(covMat)), res.t()) * 0.5
        # den = (torch.det(covMat)**0.5)
        # print(num, den)
        prb = torch.exp(-num)
        return prb

    def compute_normal_prob(self, z):
        d = self.z_dims
        pz = torch.exp(-torch.sum(z**2)*0.5) / (6.283)**(d*0.5)
        return pz

    def high_metric(self, numGen):
        thresholds = [-90, -70, -50, -30, -10, -1]
        avgVals = [0, 0 , 0, 0, 0, 0]
        allSamps = self.sample_qz(numGen)
        # muAll = np.zeros([len(self.val_loader.dataset), self.z_dims])
        # sigAll = np.zeros([len(self.val_loader.dataset), self.z_dims])
        # for i in range(len(self.val_loader.dataset)):
        #     d = self.val_loader.dataset[i]
        #     d = d.to(self.device).view(-1, 3, 64, 64)
        #     ff = self.model.encoder(d)
        #     muAll[i, :] = ff[:, :self.z_dims].cpu().detach().numpy()
        #     sigAll[i, :] = ff[:, self.z_dims:].exp().cpu().detach().numpy()
        # np.save(self.output_dir + '/muAll.npy', muAll)
        # np.save(self.output_dir + '/sigAll.npy', sigAll)
        muAll = np.load(self.output_dir + '/muAll.npy')
        sigAll = np.load(self.output_dir + '/sigAll.npy')
        visited = np.zeros([allSamps.shape[0],])
        for t in range(len(thresholds)):
            th = thresholds[t]
            print('Now at thereshold', th)
            count = 0
            for i in range(allSamps.shape[0]):
                # compute logp(z)
                z = allSamps[i, :].cpu().detach().numpy()
                if self.implicit:
                    z = torch.from_numpy(z).view(-1, self.z_dims)
                    z = z.to(self.device)
                    z0 = self.model.g(z)[0]
                    # print(z0.cpu().detach().numpy(), z.cpu().detach().numpy())
                    lpz = self.probMUltiGaussian(z0.cpu().detach().numpy()[0], np.zeros([self.z_dims,]), np.ones([self.z_dims,]))
                else:
                    lpz = self.probMUltiGaussian(z, np.zeros([self.z_dims,]), np.ones([self.z_dims,]))
                # print(lpz)
                # import pdb; pdb.set_trace()
                z = allSamps[i, :].cpu().detach().numpy()
                if lpz < th:
                    count += 1
                if lpz < th and visited[i] == 0:
                    # print("This ges thorough")
                    visited[i] = 1
                    qzslog = self.qz_prob(z, muAll, sigAll)
                    for k in range(t, 6):
                        avgVals[k] += qzslog - lpz
                    
            if count > 0:
                avgVals[t] /= count
        np.save(self.output_dir + '/checking.npy', avgVals)


    def generate_highpos_samples(self, N):
        self.model.eval()
        output_dir = osp.join(self.output_dir, str(self.global_iter))
        os.makedirs(output_dir, exist_ok=True)
        genpath = osp.join(output_dir, 'highpos')
        os.makedirs(genpath, exist_ok=True)
        count = 0
        for k in range(int(N/self.batch_size)):
            # print(k)
            allSamps = self.sample_qz(self.batch_size*10)
            if self.implicit:
                qz_samps_z0 = self.model.g(allSamps)[0]
                hi_z0, hi_lqz0 = self.high_posterior(qz_samps_z0, look_size=self.batch_size)
                hi_pzs = self.model.g_inv(hi_z0)
            else:
                hi_pzs, hi_lqzs = self.high_posterior(allSamps, look_size=self.batch_size)
            hi_pts = self.model.decoder(hi_pzs).detach()
            for i in range(self.batch_size):
                temp = hi_pts[i, ...]
                save_image(tensor=temp.cpu(), nrow=self.z_dims*hi_pts.size(0), pad_value=1,
                  filename=osp.join(genpath, 'highpos' + str(k*self.batch_size + i) + '.png')
                )


    def generate_original(self):
        self.model.eval()
        output_dir = osp.join(self.output_dir, str(self.global_iter))
        genpath = osp.join(output_dir, 'original')
        os.makedirs(genpath, exist_ok=True)
        count = 0
        for x in self.data_loader:
            for i in range(self.batch_size):
                temp = x[i, ...]
                save_image(tensor=temp.cpu(), nrow=self.z_dims*x.size(0), pad_value=1,
                filename=osp.join(genpath, 'original' + str(count*self.batch_size + i) + '.png'))
            count += 1

    def generate_lowpos_samples(self, numGen):
        self.model.eval()
        output_dir = osp.join(self.output_dir, str(self.global_iter))
        os.makedirs(output_dir, exist_ok=True)
        genpath = osp.join(output_dir, 'lowpos')
        os.makedirs(genpath, exist_ok=True)
        count = 0
        for x in self.data_loader:
            if isinstance(x, list):
                x = x[0]
            # x = torch.from_numpy(np.array(x))
            qzs = self.model.encoder(x.to(self.device))
            mus = qzs[:, :self.z_dims]
            logvars = qzs[:, self.z_dims:]
            low_pzs, low_lqzs = self.low_posterior(mus, logvars, look_size=self.batch_size)
            low_pts = self.model.decoder(low_pzs).detach()
            for i in range(self.batch_size):
                temp = low_pts[i, ...]
                save_image(tensor=temp.cpu(), nrow=self.z_dims*low_pts.size(0), pad_value=1,
                  filename=osp.join(genpath, 'lowpos' + str(count*self.batch_size + i) + '.png')
                )
            count +=1
            if count*self.batch_size > numGen - 1:
                break


    def generate_normal_samples(self, numGen):
        self.model.eval()
        output_dir = osp.join(self.output_dir, str(self.global_iter))
        os.makedirs(output_dir, exist_ok=True)
        genpath = osp.join(output_dir, 'generated')
        os.makedirs(genpath, exist_ok=True)
        N = int(numGen / self.batch_size)
        P = np.zeros([numGen, self.input_dims, self.img_size, self.img_size])
        for k in range(N):
            gen_x = self.generate_sample(self.batch_size)
            for i in range(self.batch_size):
                temp = gen_x[i, ...]
                save_image(tensor=temp.cpu(), nrow=self.z_dims*gen_x.size(0), pad_value=1,
                filename=osp.join(genpath, 'generated' + str(k*self.batch_size + i) + '.png'))

    def draw_generated(self):
        output_dir = osp.join(self.output_dir, str(self.global_iter))
        os.makedirs(output_dir, exist_ok=True)
        gen_x = self.generate_sample(30)
        # gen_grid = make_grid(gen_x)
        save_image(tensor=gen_x.cpu(), nrow=6, padding=0,
          filename=osp.join(output_dir, 'generated.png')
        )


    def linear_generated(self):
        output_dir = osp.join(self.output_dir, str(self.global_iter))
        os.makedirs(output_dir, exist_ok=True)
        gen_x = self.generate_sample(self.batch_size)
        gen_x = gen_x.cpu().data.numpy()
        plt.scatter(gen_x[:,0], gen_x[:,1])
        plt.savefig(osp.join(output_dir, 'generated.png'))
        plt.close()


    def generate_sample(self, sample_size):
        self.model.eval()
        z = self.prior.sample((sample_size,)).to(self.device)
        if self.implicit:
            z = self.model.g_inv(z)
        out_sample = self.model.decoder(z)
        self.model.train()
        return out_sample

    def computeGenMetrics(self):
        # first compute the NLL
        with torch.no_grad():
            self.model.eval()
            ELBO = 0
            count = 0
            for x in self.val_loader:
                count+=1
                if self.implicit:
                    x_recon, mu, logvar, z, g_1, g_2 = self.model(x.to(self.device))
                    # likelihood loss
                    recon_loss = self.reconstruction_loss(x.to(self.device), x_recon)
                    # DKL loss
                    ELS = -1*torch.sum(g_2)
                    k = torch.mul(g_1, g_1).sum(dim=1)
                    DKL = -0.5*torch.sum(logvar.sum(dim=1) - k)
                    kld = ELS+DKL

                else:
                    x_recon, mu, logvar, z = self.model(x.to(self.device))
                    recon_loss = self.reconstruction_loss(x.to(self.device), x_recon)
                    kld = self.kl_divergence(mu, logvar)
                ELBO += (recon_loss + kld).mean()
            NLL = (ELBO/count)
            NLL = NLL.cpu().detach().numpy()


            samplesize = 5000
            # now the sym KL divergence
            if self.implicit:
                pz = self.prior.sample((samplesize,))
                qz_samps = self.sample_qz(sample_size=samplesize)
                qz_samps_z0 = self.model.g(qz_samps)[0]
                pz = pz.cpu().detach().numpy()
                qz_samps_z0 = qz_samps_z0.cpu().detach().numpy()
                KLDiv = KLdivergence(pz, qz_samps_z0) + KLdivergence(qz_samps_z0, pz)
            else:
                pz = self.prior.sample((samplesize,))
                qz_samps = self.sample_qz(sample_size=samplesize)
                pz = pz.cpu().detach().numpy()
                qz_samps = qz_samps.cpu().detach().numpy()
                KLDiv = KLdivergence(pz, qz_samps) + KLdivergence(qz_samps, pz)
            # KLDiv = 0
            output_dir = osp.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            d = np.array([NLL/self.mseWeight/1000, KLDiv])
            np.save(output_dir + '/genMetrics.npy', d)

            self.model.train()

    def low_posterior(self, mus=None, logvars=None, z_samples=500, look_size=100):
        self.model.eval()
        if mus is None or logvars is None:
            qzs = torch.cat(
              [self.model.encoder(x.to(self.device)) for x in self.val_loader], dim=0
            )
            mus = qzs[:, :self.z_dims]
            logvars = qzs[:, self.z_dims:]
        pzs, logqzs = [], []
        for i in range(z_samples):
            pz = self.prior.sample().to(self.device)
            if self.implicit:
                pz = self.model.g_inv(pz.to(self.device)).detach()
            log_mean_exp = (
              -1*((pz[0]-mus[:,0]).pow(2)/logvars[:,0] + (pz[1]-mus[:,1]).pow(2)/logvars[:,1])
            ).exp().mean()
            log_qz = torch.log( log_mean_exp )
            pzs.append( pz.detach().unsqueeze(0) )
            logqzs.append( log_qz.detach().unsqueeze(0) )
        pzs = torch.cat(pzs)
        logqzs = torch.cat(logqzs)
        sidx = torch.argsort(logqzs)
        pzs = pzs[sidx]
        logqzs = logqzs[sidx]
        self.model.train()
        return pzs[:look_size], logqzs[:look_size] # these are the low posterior samples that we care about

    def high_posterior(self, zs, look_size=100):
        log_mean_exp = (
          -1*((zs[:,0]).pow(2) + (zs[:,1]).pow(2))
        ).exp()
        sidx = torch.argsort(log_mean_exp)
        lqz = log_mean_exp[sidx]
        zs = zs[sidx]
        return zs[:look_size], lqz[:look_size]

    def draw_qz(self):
        output_dir = osp.join(self.output_dir, str(self.global_iter))
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()
        if self.img_size > 28:
            for x in self.val_loader:
                qzs = self.model.encoder(x.to(self.device))
                break
        else:
            qzs = torch.cat(
              [self.model.encoder(x.to(self.device)) for x in self.val_loader], dim=0
            )

        mus = qzs[:, :self.z_dims]
        logvars = qzs[:, self.z_dims:]
        qzs = self.model.reparametrize(mus, logvars).detach()
        if self.data_type == 'linear':
            recon = self.model.decoder(qzs.to(self.device)).detach().cpu().numpy()
        self.draw_qzs(qzs)
        sample_size = 5000 if self.data_type == 'linear' else self.batch_size
        if self.img_size > 28:
            qz_samps = self.sample_qz(sample_size=sample_size)
        else:
            qz_samps = self.sample_qz(sample_size=sample_size, qzs=qzs)
        low_pzs, low_lqzs = self.low_posterior(mus, logvars, look_size=self.batch_size)
        if self.implicit:
            qz_samps_z0 = self.model.g(qz_samps)[0]
            hi_z0, hi_lqz0 = self.high_posterior(qz_samps_z0)
            hi_pzs = self.model.g_inv(hi_z0)
        else:
            hi_pzs, hi_lqzs = self.high_posterior(qz_samps)
        qz_sampsn = qz_samps.cpu().detach().numpy()
        low_pzsn = low_pzs.cpu().detach().numpy()
        hi_pzsn = hi_pzs.cpu().detach().numpy()
        np.save(output_dir + '/qz_samples.npy', qz_sampsn)
        np.save(output_dir + '/high_pos_z.npy', hi_pzsn)
        np.save(output_dir + '/low_pos_z.npy', low_pzsn)

        if self.implicit:
            low_z0 = self.model.g(low_pzs)[0].detach().cpu().numpy()
            z0_samples = self.model.g(qz_samps)[0].detach().cpu().numpy()
            np.save(output_dir + '/qz0_samples.npy', z0_samples)
            np.save(output_dir + '/high_pos_z0.npy', hi_z0.cpu().detach().numpy())
            np.save(output_dir + '/low_pos_z0.npy', low_z0)

        low_pts = self.model.decoder(low_pzs).detach()
        hi_pts = self.model.decoder(hi_pzs).detach()

        if self.data_type == 'linear':
            low_pts = low_pts.cpu().numpy()
            hi_pts = hi_pts.cpu().numpy()
            orig = torch.cat( [x for x in self.val_loader], dim=0 )
            np.save(output_dir + '/original.npy', orig)
            np.save(output_dir + '/high_pos_x.npy', hi_pts)
            np.save(output_dir + '/low_pos_x.npy', low_pts)
        else:
            self.highLow_draw_reconstruction(
              low_pts[:30], hi_pts[:30], 'low_post.png', 'hi_post.png'
            )
        plt.close()
        self.model.train()

    def draw_qzs(self, qzs=None):
        if self.z_dims != 2:
            return
        output_dir = osp.join(self.output_dir, str(self.global_iter))
        os.makedirs(output_dir, exist_ok=True)
        if qzs is None:
            qzs = torch.cat(
              [self.model.encoder(x.to(self.device)) for x in self.val_loader], dim=0
            )
            mus = qzs[:, :self.z_dims]
            logvars = qzs[:, self.z_dims:]
            qzs = self.model.reparametrize(mus, logvars)
        z = qzs.detach().cpu().numpy()
        plt.scatter(z[:,0], z[:,1])
        plt.savefig(osp.join(output_dir, 'qz_points.png'))
        plt.close()

    def sample_gen_highpos(self, sample_size):
        sample_size = sample_size*4
        N = len(self.val_loader.dataset)
        samples = np.zeros([sample_size, self.z_dims])
        probs = np.zeros([sample_size, ])
        mus = torch.zeros([N, self.z_dims])
        logvars = torch.zeros([N, self.z_dims])
        for i in range(N):
            datain = self.val_loader.dataset[i]
            datain = datain.view(-1, 1, 28, 28)
            # print(datain.shape)
            # datain = torch.from_numpy(datain)
            datain = datain.to(self.device)
            kk = self.model.encoder(datain)
            mus[i, :] = kk[:, :self.z_dims]
            logvars[i, :] = kk[:, self.z_dims:]
        count = 0
        for i in range(sample_size):
            k = np.random.random_integers(0, N-1)
            d = self.val_loader.dataset[k]
            d = d.view(-1,1, 28, 28)
            x = d.to(self.device)
            qzs = self.model.encoder(x)
            z = self.model.reparametrize(qzs[:, :self.z_dims], qzs[:, self.z_dims:])
            samples[i, ...] = z.cpu().detach().numpy() 
            tempprob = (-1*((z[0]-mus[0, ...]).pow(2)/logvars[0, ...] + (z[1]-mus[1, ...]).pow(2)/logvars[1, ...])).exp().mean()
            probs[i] = tempprob.cpu().detach().numpy()
            sidx = np.argsort(probs)

        return samples[sidx[int(0.75*sample_size):], ...]

    def sample_qz(self, sample_size=1000, qzs=None):
        N = len(self.val_loader.dataset)
        if qzs is not None:
            N = qzs.size(0)
            ks = np.random.randint(0, N, sample_size)
            z = qzs[ks]
            return z

        ks = np.random.randint(0, N, sample_size)
        imps = torch.cat(
          [self.val_loader.dataset[k].unsqueeze(0) for k in ks]
        )
        # imps = self.val_loader.dataset[ks]
        qzs = self.model.encoder(imps.to(self.device))
        mus = qzs[:, :self.z_dims]
        logvars = qzs[:, self.z_dims:]
        z = self.model.reparametrize(mus, logvars)
        return z.detach()

    def draw_loss(self):
        output_dir = osp.join(self.output_dir, str(self.global_iter))
        os.makedirs(output_dir, exist_ok=True)

        for key in self.meta:
            if isinstance(self.meta[key], list):
                if isinstance(self.meta[key][0], float):
                    plt.plot( self.meta[key] )
                    plt.savefig( osp.join(output_dir, '{}.png'.format(key)) )
                    plt.clf()
                    temp = np.array(self.meta[key])
                    np.save(osp.join(output_dir, '{}.npy'.format(key)), temp)
                else:
                    val = torch.stack(self.meta[key]).cpu().numpy()
                    plt.imshow(val)
                    plt.savefig(osp.join(output_dir, 'mat_{}.png'.format(key)))
                    plt.clf()
            elif isinstance(self.meta[key], tuple) and len(self.meta[key]) == 2:
                plt.plot( self.meta[key][0], self.meta[key][1] )
                plt.savefig( osp.join(output_dir, '{}.png'.format(key)) )
                plt.clf()
                temp = np.array(self.meta[key])
                np.save(osp.join(output_dir, '{}.npy'.format(key)), temp)
        plt.close()


    def linear_traverse(self, limit=3, inter=2/3, loc=-1):
        pass

    def visualize_traverse(self, limit=3, inter=2/3, loc=-1):
        self.model.eval()
        output_dir = osp.join(self.output_dir, str(self.global_iter))
        os.makedirs(output_dir, exist_ok=True)
        import random

        decoder = self.model.decoder
        encoder = self.model.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)

        n_dsets = len(self.val_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img = self.val_loader.dataset.__getitem__(rand_idx)
        if isinstance(random_img, (list, tuple)):
            random_img = random_img[0]
        random_img = random_img.to(self.device).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dims]

        random_z = torch.rand(1, self.z_dims).to(self.device)

        fixed_idx = 0
        fixed_img = self.val_loader.dataset.__getitem__(fixed_idx)
        if isinstance(fixed_img, (list, tuple)):
            fixed_img = fixed_img[0]
        fixed_img = fixed_img.to(self.device).unsqueeze(0)
        fixed_img_z = encoder(fixed_img)[:, :self.z_dims]

        Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}
        gifs = []
        for key in Z:
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dims):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = torch.sigmoid(decoder(z).detach()).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            save_image(tensor=samples.cpu(), nrow=self.z_dims, pad_value=1,
              filename=osp.join(output_dir, '{}.png'.format(key))
            )
        self.model.train()

    def linear_traversal(self, limit=3, inter=2/3, loc=-1):
        pass

    def traversal(self, num_inters=100, iterp_steps=10):
        self.model.eval()
        self.iterp_steps = iterp_steps
        output_dir = osp.join(self.output_dir, str(self.global_iter), 'traversals')
        os.makedirs(output_dir, exist_ok=True)
        bestf_dir = osp.join(output_dir, 'best_factor')
        os.makedirs(bestf_dir, exist_ok=True)

        rankings, stdvs = self.rank_zdims()
        # find batch_size best
        for xi, x in enumerate(self.val_loader):
            x = x.to(self.device)
            x_recon = self.model(x)[0].detach()
            # recon_loss = F.mse_loss( x, x_recon, reduction=None )
            recon_loss = (x - x_recon).pow(2).sum(-1).sum(-1).sum(-1).sqrt()
            best_3 = torch.argsort(recon_loss)[:3]
            for xb in best_3:
                self.traverse_hiz(x[xb],
                  osp.join(bestf_dir, 'factors_{}'.format(xb.cpu().item())),
                  rankings=rankings, stdvs=stdvs
                )
            # self.traverse_all(x[best_3[0]], x[best_3[1]],
            #   osp.join(output_dir, 'traverseall_{}_{}.png'.format(fix[0], fix[1]))
            # )
            if xi > 5:
                break

        if self.z_dims > 2:
            # (CelebA_Cropped_Val/187_celeba_31.png, CelebA_Cropped_Val/193_celeba_662.png)
            # (CelebA_Cropped_Val/186_celeba_277.png, CelebA_Cropped_Val/198_celeba_994.png)
            # (CelebA_Cropped_Val/186_celeba_530.png, CelebA_Cropped_Val/192_celeba_233.png)
            fixed = [(10879, 5577), (15492, 8308), (9542, 12632)]
        else:
            fixed = [(484, 2234), (1198, 4521), (989,6043), (292, 5624), (5678, 8900)]
            Dl = len(self.val_loader.dataset)
            Z = np.zeros([Dl, 2])
            if self.implicit:
                Z0 = np.zeros([Dl, 2])
            count = 0
            for x in self.val_loader:
                temp = self.model.encoder(x.to(self.device))
                mu = temp[:, :self.z_dims]
                Z[count*self.batch_size:(count + 1)*self.batch_size, :] = mu.cpu().detach().numpy()
                if self.implicit:
                    [muz, vv] = self.model.g(mu)
                    Z0[count*self.batch_size:(count + 1)*self.batch_size, :] = muz.cpu().detach().numpy()
                count += 1
            np.save(output_dir + '/Zsamples.npy', Z)
            if self.implicit:
                np.save(output_dir + '/Z0samples.npy', Z0)
            # np.save(output_dir + '/fixedIndex.npy', fixed)

        for fix in fixed:
            rimg1 = self.val_loader.dataset.__getitem__( fix[0] )
            rimg2 = self.val_loader.dataset.__getitem__( fix[1] )
            if isinstance(rimg1, (list, tuple)):
                rimg1 = rimg1[0]
                rimg2 = rimg2[0]
            self.traverse_all(rimg1, rimg2,
              osp.join(output_dir, 'traverseall_{}_{}.png'.format(fix[0], fix[1]))
            )
            # self.traverse_z(rimg1, rimg2,
            #   osp.join(output_dir, 'traversez_{}_{}.png'.format(fix[0], fix[1]))
            # )
            # self.traverse_hiz(rimg1,
            #   osp.join(bestf_dir, 'factors_{}'.format(fix[0])),
            #   rankings=rankings, stdvs=stdvs
            # )
            # self.traverse_hiz(rimg2,
            #   osp.join(bestf_dir, 'factors_{}'.format(fix[1])),
            #   rankings=rankings, stdvs=stdvs
            # )

        randpath = 'traverse_indx.npy' # osp.join(output_dir, 'traverse_indx.npy')
        if os.path.exists(randpath):
            rands = np.load(randpath)
        else:
            rands = np.random.randint(len(self.val_loader.dataset), size=(num_inters, 2))
        for rand in rands:
            rimg1 = self.val_loader.dataset.__getitem__( rand[0] )
            rimg2 = self.val_loader.dataset.__getitem__( rand[1] )
            if isinstance(rimg1, (list, tuple)):
                rimg1 = rimg1[0]
                rimg2 = rimg2[0]

            self.traverse_all(rimg1, rimg2,
              osp.join(output_dir, 'traverseall_{}_{}.png'.format(rand[0], rand[1]))
            )
            # self.traverse_z(rimg1, rimg2,
            #   osp.join(output_dir, 'traversez_{}_{}.png'.format(rand[0], rand[1]))
            # )
        np.save( randpath, rands )

        self.model.train()

    def rank_zdims(self):
        self.model.eval()
        samplesize = 10000 // self.batch_size
        amu = torch.zeros(self.z_dims).to(self.device)
        for xi, x in enumerate(self.val_loader):
            if xi >= samplesize:
                break
            mu = self.model.encoder(x.to(self.device))[:, :self.z_dims]
            if self.implicit:
                mu = self.model.g(mu)[0]
            # print(mu.shape, amu.shape, mu.sum(1).detach().shape)
            amu += mu.sum(0).detach()
        stdv = torch.zeros(self.z_dims).to(self.device)
        for xi, x in enumerate(self.val_loader):
            if xi >= samplesize:
                break
            mu = self.model.encoder(x.to(self.device))[:, :self.z_dims]
            if self.implicit:
                mu = self.model.g(mu)[0]
            stdv += (mu-amu).pow(2).sum(0).detach() / self.batch_size
        stdv = (stdv/samplesize).sqrt()
        std_idx = torch.argsort(stdv, descending=True)
        print('most influential dimension:', std_idx[0].item())
        self.model.train()
        return std_idx, stdv

    def traverse_hiz(self, img, outfile, rankings=None, stdvs=None):
        if rankings is None:
            rankings, stdvs = self.rank_zdims()
        iterp_steps = 6
        variation = 5
        img = img.to(self.device).unsqueeze(0)
        img_z = self.model.encoder(img)[:, :self.z_dims]
        if self.implicit:
            img_z = self.model.g(img_z)[0]
        img_z = img_z.detach()
        intrp_imgs = [] #[img]
        zdims = []
        for zdim in range(np.min([5, self.z_dims])):
            zdim = rankings[zdim]
            zdims.append(str(zdim.cpu().item()))
            timg_z = img_z.clone()
            beg, end = timg_z[:,zdim]-variation, timg_z[:,zdim]+variation
            istep = (beg - end).abs()/iterp_steps
            for j in range(0, iterp_steps+1):
                timg_z[:,zdim] = beg + (j*istep)
                if self.implicit:
                    timg = self.model.decoder(self.model.g_inv(timg_z)).detach()
                else:
                    timg = self.model.decoder(timg_z).detach()
                intrp_imgs.append(timg)
        intrp_imgs = torch.cat(intrp_imgs)
        save_image(tensor=intrp_imgs.cpu(), nrow=iterp_steps+1, pad_value=1,
          filename=outfile + '_{}.png'.format('_'.join(zdims))
        )

    def traverse_all(self, rimg1, rimg2, outfile, savenpy=False):
        iterp_steps = self.iterp_steps
        decoder = self.model.decoder
        encoder = self.model.encoder
        rimg1 = rimg1.to(self.device).unsqueeze(0)
        rimg2 = rimg2.to(self.device).unsqueeze(0)
        output_dir = osp.join(self.output_dir, str(self.global_iter), 'traversals')
        rimgs = torch.cat( [rimg1, rimg2], dim=0 )
        rimg_z = encoder(rimgs)[:, :self.z_dims]
        tptsz = []
        spt = outfile.rsplit('/', 1)
        nm = spt[1]
        # np.save(output_dir + '/' + nm + 'Z.npy', rimg_z.cpu().detach().numpy())
        if self.implicit:
            rimg_z = self.model.g(rimg_z)[0]
            tptsz0 = []
            # np.save(output_dir + '/' + nm + 'Z0.npy', rimg_z.cpu().detach().numpy())
        rimg_z1, rimg_z2 = rimg_z.detach()

        # steps along all z dims
        istep = (rimg_z1 - rimg_z2).abs()/iterp_steps
        istep[(rimg_z1 - rimg_z2)>=0] *= -1
        # intrp_imgs = [rimg1]
        intrp_imgs = []

        for j in range(0, iterp_steps+1):
            timg_z = rimg_z1 + (j*istep)
            if self.implicit:
                tptsz0.append(timg_z)
                timg_z = self.model.g_inv(timg_z)
            tptsz.append(timg_z)
            timg = decoder(timg_z).detach()
            intrp_imgs.append(timg)
        # intrp_imgs.append(rimg2)
        intrp_imgs = torch.cat(intrp_imgs)
        tpts = torch.stack(tptsz)
        if savenpy:
            np.save(output_dir + '/' + nm + 'Z.npy', tpts.cpu().detach().numpy())
            if self.implicit:
                tpts = torch.stack(tptsz0)
                np.save(output_dir + '/' + nm + 'Z0.npy', tpts.cpu().detach().numpy())
        save_image(tensor=intrp_imgs.cpu(), nrow=iterp_steps+1, pad_value=1,
          filename=outfile
        )

    def traverse_z(self, rimg1, rimg2, outfile):
        iterp_steps = self.iterp_steps
        decoder = self.model.decoder
        encoder = self.model.encoder
        rimg1 = rimg1.to(self.device).unsqueeze(0)
        rimg2 = rimg2.to(self.device).unsqueeze(0)
        rimgs = torch.cat( [rimg1, rimg2], dim=0 )
        rimg_z = encoder(rimgs)[:, :self.z_dims]
        if self.implicit:
            rimg_z = self.model.g(rimg_z)[0]
        rimg_z1, rimg_z2 = rimg_z.detach()

        # steps for each z dim
        intrp_imgs = [rimg1]
        timg_z = rimg_z1
        if self.implicit:
            timg = decoder(self.model.g_inv(timg_z)).detach()
        else:
            timg = decoder(timg_z).detach()
        intrp_imgs.append(timg)
        for z in range(self.z_dims):
            timg_z[z] = rimg_z2[z]
            if self.implicit:
                timg = decoder(self.model.g_inv(timg_z)).detach()
            else:
                timg = decoder(timg_z).detach()
            intrp_imgs.append(timg)
        intrp_imgs.append(rimg2)
        intrp_imgs = torch.cat(intrp_imgs)
        save_image(tensor=intrp_imgs.cpu(), nrow=iterp_steps+1, pad_value=1,
          filename=outfile
        )

    # model save functions

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.model.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        states = {'iter':self.global_iter,
                  'meta':self.meta,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = osp.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        try:
            file_path = osp.join(self.ckpt_dir, filename)
            if osp.isfile(file_path):
                checkpoint = torch.load(file_path)
                self.global_iter = checkpoint['iter']
                self.meta = checkpoint['meta']
                self.model.load_state_dict(checkpoint['model_states']['net'])
                self.optim.load_state_dict(checkpoint['optim_states']['optim'])
                print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
            else:
                print("=> no checkpoint found at '{}'".format(file_path))
        except Exception as e:
            print('model couldnt load')
