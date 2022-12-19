## --------------------------------------------------------------------------
## Saliency in Augmented Reality
## Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
## ACM International Conference on Multimedia (ACM MM 2022)
## --------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init



def nss(pred, fixations):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    fixations = fixations.reshape(new_size)

    pred_normed = (pred - pred.mean(-1, True)) / pred.std(-1, keepdim=True)
    results = []
    for this_pred_normed, mask in zip(torch.unbind(pred_normed, 0),
                                      torch.unbind(fixations, 0)):
        if mask.sum() == 0:
            print("No fixations.")
            results.append(torch.ones([]).float().to(fixations.device))
            continue
        nss_ = torch.masked_select(this_pred_normed, mask)
        nss_ = nss_.mean(-1)
        results.append(nss_)
    results = torch.stack(results)
    results = results.reshape(size[:2])
    return results


def corr_coeff(pred, target):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    target = target.reshape(new_size)

    cc = []
    for x, y in zip(torch.unbind(pred, 0), torch.unbind(target, 0)):
        xm, ym = x - x.mean(), y - y.mean()
        r_num = torch.mean(xm * ym)
        r_den = torch.sqrt(
            torch.mean(torch.pow(xm, 2)) * torch.mean(torch.pow(ym, 2)))
        r = r_num / r_den
        cc.append(r)

    cc = torch.stack(cc)
    cc = cc.reshape(size[:2])
    return cc  # 1 - torch.square(r)


def kld_loss(pred, target):
    loss = F.kl_div(pred, target, reduction='none')
    loss = loss.sum(-1).sum(-1).sum(-1)
    return loss






class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class SalLoss(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

        self.CC = CCLoss()
        self.KL = KLLoss()
        self.NSS = NSSLoss()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, reconstructions, gts, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        
        rec_loss = torch.abs(gts.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(gts.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # saliency loss
        # CC_loss = corr_coeff(reconstructions,gts)
        # KL_loss = kld_loss(reconstructions,gts)
        CC_loss = self.CC(reconstructions,gts)
        KL_loss = self.KL(reconstructions,gts)
        NSS_loss = self.NSS(reconstructions,gts)

        # paired image-to-image translation loss (with condition):
        # ########################################################
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean() + 0.2*(1+CC_loss+KL_loss)

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/CC_loss".format(split): CC_loss.detach().mean(),
                   "{}/KL_loss".format(split): KL_loss.detach().mean(),
                   "{}/NSS_loss".format(split): NSS_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(gts.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((gts.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log





class CCLoss(nn.Module):
    def __init__(self):
        super(CCLoss, self).__init__()        
        # self.vgg = Vgg19().cuda()
        # self.criterion = nn.L1Loss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] 
        self.epsilon = 1e-8


    def forward(self, map_pred, map_gtd):  # map_pred : input prediction saliency map, map_gtd : input ground truth density map
        map_pred = map_pred.float()
        map_gtd = map_gtd.float()
        
        map_pred = map_pred.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols
        map_gtd = map_gtd.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols

        min1 = torch.min(map_pred)
        max1 = torch.max(map_pred)
        map_pred = (map_pred - min1) / (max1 - min1 + self.epsilon) # min-max normalization for keeping KL loss non-NAN

        min2 = torch.min(map_gtd)
        max2 = torch.max(map_gtd)
        map_gtd = (map_gtd - min2) / (max2 - min2 + self.epsilon) # min-max normalization for keeping KL loss non-NAN
        
        map_pred_mean = torch.mean(map_pred) # calculating the mean value of tensor
        map_pred_mean = map_pred_mean.item() # change the tensor into a number

        map_gtd_mean = torch.mean(map_gtd) # calculating the mean value of tensor
        map_gtd_mean = map_gtd_mean.item() # change the tensor into a number
        # print("map_gtd_mean is :", map_gtd_mean)

        map_pred_std = torch.std(map_pred) # calculate the standard deviation
        map_pred_std = map_pred_std.item() # change the tensor into a number 
        map_gtd_std = torch.std(map_gtd) # calculate the standard deviation
        map_gtd_std = map_gtd_std.item() # change the tensor into a number 

        map_pred = (map_pred - map_pred_mean) / (map_pred_std + self.epsilon) # normalization
        map_gtd = (map_gtd - map_gtd_mean) / (map_gtd_std + self.epsilon) # normalization

        map_pred_mean = torch.mean(map_pred) # re-calculating the mean value of normalized tensor
        map_pred_mean = map_pred_mean.item() # change the tensor into a number

        map_gtd_mean = torch.mean(map_gtd) # re-calculating the mean value of normalized tensor
        map_gtd_mean = map_gtd_mean.item() # change the tensor into a number

        CC_1 = torch.sum( (map_pred - map_pred_mean) * (map_gtd - map_gtd_mean) )
        CC_2 = torch.rsqrt(torch.sum(torch.pow(map_pred - map_pred_mean, 2))) * torch.rsqrt(torch.sum(torch.pow(map_gtd - map_gtd_mean, 2))) + self.epsilon
        CC = CC_1 * CC_2
        # print("CC loss is :", CC)
        CC = -CC # the bigger CC, the better



        # we put the L1 loss with CC together for avoiding building a new class
        # L1_loss =  torch.mean( torch.abs(map_pred - map_gtd) )
        # print("CC and L1 are :", CC, L1_loss)
        # CC = CC + L1_loss

        return CC


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()        
        # self.vgg = Vgg19().cuda()
        # self.criterion = nn.L1Loss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] 
        self.epsilon = 1e-8 # the parameter to make sure the denominator non-zero


    def forward(self, map_pred, map_gtd):  # map_pred : input prediction saliency map, map_gtd : input ground truth density map
        map_pred = map_pred.float()
        map_gtd = map_gtd.float()
        
        map_pred = map_pred.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols
        map_gtd = map_gtd.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols

        min1 = torch.min(map_pred)
        max1 = torch.max(map_pred)
        # print("min1 and max1 are :", min1, max1)
        map_pred = (map_pred - min1) / (max1 - min1 + self.epsilon) # min-max normalization for keeping KL loss non-NAN

        min2 = torch.min(map_gtd)
        max2 = torch.max(map_gtd)
        # print("min2 and max2 are :", min2, max2)
        map_gtd = (map_gtd - min2) / (max2 - min2 + self.epsilon) # min-max normalization for keeping KL loss non-NAN

        map_pred = map_pred / (torch.sum(map_pred) + self.epsilon)# normalization step to make sure that the map_pred sum to 1
        map_gtd = map_gtd / (torch.sum(map_gtd) + self.epsilon) # normalization step to make sure that the map_gtd sum to 1
        # print("map_pred is :", map_pred)
        # print("map_gtd is :", map_gtd)


        KL = torch.log(map_gtd / (map_pred + self.epsilon) + self.epsilon)
        # print("KL 1 is :", KL)
        KL = map_gtd * KL
        # print("KL 2 is :", KL)
        KL = torch.sum(KL)
        # print("KL 3 is :", KL)
        # print("KL loss is :", KL)

        return KL

class NSSLoss(nn.Module):
    def __init__(self):
        super(NSSLoss, self).__init__()        
        # self.vgg = Vgg19().cuda()
        # self.criterion = nn.L1Loss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] 
        self.epsilon = 1e-8 # the parameter to make sure the denominator non-zero


    def forward(self, map_pred, map_gtd):  # map_pred : input prediction saliency map, map_gtd : input ground truth density map
        map_pred = map_pred.float()
        map_gtd = map_gtd.float()

        map_pred = map_pred.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols
        map_gtd = map_gtd.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols

        min1 = torch.min(map_pred)
        max1 = torch.max(map_pred)
        # print("min1 and max1 are (saliecny map):", min1, max1)
        map_pred = (map_pred - min1) / (max1 - min1 + self.epsilon) # min-max normalization for keeping KL loss non-NAN

        min2 = torch.min(map_gtd)
        max2 = torch.max(map_gtd)
        # print("min2 and max2 are (fixation points) :", min2, max2)
        map_gtd = (map_gtd - min2) / (max2 - min2 + self.epsilon) # min-max normalization for keeping KL loss non-NAN
        
        map_gtd_id_1 = torch.gt(map_gtd, 0.5)
        map_gtd_id_0 = torch.lt(map_gtd, 0.5)
        map_gtd_id_00 = torch.eq(map_gtd, 0.5)
        map_gtd[map_gtd_id_1] = 1.0
        map_gtd[map_gtd_id_0] = 0.0
        map_gtd[map_gtd_id_00] = 0.0

        map_pred_mean = torch.mean(map_pred) # calculating the mean value of tensor
        map_pred_mean = map_pred_mean.item() # change the tensor into a number

        map_pred_std = torch.std(map_pred) # calculate the standard deviation
        map_pred_std = map_pred_std.item() # change the tensor into a number 

        map_pred = (map_pred - map_pred_mean) / (map_pred_std + self.epsilon) # normalization

        NSS = map_pred * map_gtd
        # print("early NSS is :", NSS)
        '''
        dim_NSS = NSS.size()
        print("dim_NSS is :", dim_NSS)
        dim_NSS = dim_NSS[1]
        sum_nss = 0
        dim_sum = 0
        
        for idxnss in range(0, dim_NSS):
            if (NSS[0, idxnss] > 0.05): # # should not be 0, because there are a lot of 0.00XXX in map1_NSS due to float format
                sum_nss += NSS[0, idxnss]
                dim_sum += 1
        
        NSS = sum_nss / dim_sum
        '''
        # NSS = NSS # should not add anythin, because there are a lot of 0.00XXX in map1_NSS due to float format
        # id = torch.nonzero(NSS)
        id = torch.gt(NSS, 0.1) # find out the id of NSS > 0.1
        bignss = NSS[id]
        # print(bignss)
        if(len(bignss) == 0): # NSS[id] is empty 
            id = torch.gt(NSS, -0.00000001) # decrease the threshold, because must set it as tensor not inter
            bignss = NSS[id]
        # NSS = torch.sum(NSS[id])
        # NSS = torch.mean(NSS)
        NSS = torch.mean(bignss)
        
        NSS = -NSS # the bigger NSS the better
        return NSS 
        # return 0 # if return, error : TypeError: mean(): argument 'input' (position 1) must be Tensor, not float