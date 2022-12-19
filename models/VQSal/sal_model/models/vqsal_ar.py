## --------------------------------------------------------------------------
## Saliency in Augmented Reality
## Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
## ACM International Conference on Multimedia (ACM MM 2022)
## --------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main_transfer import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder, VUNet
from taming.modules.vqvae.quantize import VectorQuantizer


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        if self.image_key == "saliency":
            x = x*2.0 - 1.0
            xrec = xrec*2.0 - 1.0
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x



class VQSalModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image", image_key_AR="image_AR", image_key_BG="image_BG",
                 colorize_nlabels=None,
                 monitor=None,
                 gt_key="saliency", # ground truth key
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim, #ckpt_path=ckpt_path,
                         ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels, monitor=monitor)
        self.gt_key = gt_key    # ground truth key
        self.image_key_AR = image_key_AR    # ground truth key
        self.image_key_BG = image_key_BG    # ground truth key
        # self.sal_conv = torch.nn.Conv2d(ddconfig["out_ch"], 1, 1)   # add a conv layer for saliency prediction
        self.sal_conv = torch.nn.Conv2d(ddconfig["out_ch"], 1, kernel_size=3, stride=1, padding=1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.decoder_AR = Decoder(**ddconfig)
        self.decoder_BG = Decoder(**ddconfig)
        self.sal_conv_AR = torch.nn.Conv2d(ddconfig["out_ch"], 1, kernel_size=3, stride=1, padding=1)
        self.sal_conv_BG = torch.nn.Conv2d(ddconfig["out_ch"], 1, kernel_size=3, stride=1, padding=1)
        self.sal_conv_final = torch.nn.Conv2d(ddconfig["out_ch"], 1, kernel_size=3, stride=1, padding=1)

        self.decoder_AR.load_state_dict(self.decoder.state_dict())
        self.decoder_BG.load_state_dict(self.decoder.state_dict())
        self.sal_conv_AR.load_state_dict(self.sal_conv.state_dict())
        self.sal_conv_BG.load_state_dict(self.sal_conv.state_dict())
        self.sal_conv_final.load_state_dict(self.sal_conv.state_dict())

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input, input_AR, input_BG):
        quant, diff, _ = self.encode(input)
        quant_AR, diff_AR, _ = self.encode(input_AR)
        quant_BG, diff_BG, _ = self.encode(input_BG)
        dec = self.decoder(self.post_quant_conv(quant))
        dec = self.sal_conv(dec)
        dec_AR = self.decoder_AR(self.post_quant_conv(quant_AR))
        dec_AR = self.sal_conv_AR(dec_AR)
        dec_BG = self.decoder_BG(self.post_quant_conv(quant_BG))
        dec_BG = self.sal_conv_AR(dec_BG)
        dec = self.sal_conv_final(torch.cat((dec_AR, dec, dec_BG), 1))
        return dec, diff

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        x_AR = self.get_input(batch, self.image_key_AR)
        x_BG = self.get_input(batch, self.image_key_BG)
        gt = self.get_input(batch, self.gt_key)
        xrec, qloss = self(x, x_AR, x_BG)

        if optimizer_idx == 0:
            # autoencode
            # aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
            #                                 last_layer=self.get_last_layer(), split="train")
            aeloss, log_dict_ae = self.loss(qloss, xrec, gt, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", cond=x)

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            # discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
            #                                 last_layer=self.get_last_layer(), split="train")
            discloss, log_dict_disc = self.loss(qloss, xrec, gt, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", cond=x)
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        x_AR = self.get_input(batch, self.image_key_AR)
        x_BG = self.get_input(batch, self.image_key_BG)
        gt = self.get_input(batch, self.gt_key)
        xrec, qloss = self(x, x_AR, x_BG)

        # aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
        #                                     last_layer=self.get_last_layer(), split="val")
        aeloss, log_dict_ae = self.loss(qloss, xrec, gt, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val", cond=x)

        # discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
        #                                     last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(qloss, xrec, gt, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val", cond=x)
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        # opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
        #                           list(self.decoder.parameters())+
        #                           list(self.quantize.parameters())+
        #                           list(self.quant_conv.parameters())+
        #                           list(self.post_quant_conv.parameters()),
        #                           lr=lr, betas=(0.5, 0.9))
        opt_ae = torch.optim.Adam(list(self.decoder.parameters())+
                                  list(self.sal_conv.parameters())+
                                  list(self.decoder_AR.parameters())+
                                  list(self.sal_conv_AR.parameters())+
                                  list(self.decoder_BG.parameters())+
                                  list(self.sal_conv_BG.parameters())+
                                  list(self.sal_conv_final.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        # return self.decoder.conv_out.weight
        return self.sal_conv.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x_AR = self.get_input(batch, self.image_key_AR)
        x_BG = self.get_input(batch, self.image_key_BG)
        gt = self.get_input(batch, self.gt_key)
        x = x.to(self.device)
        x_AR = x_AR.to(self.device)
        x_BG = x_BG.to(self.device)
        gt = gt.to(self.device)
        # forward
        xrec, _ = self(x, x_AR, x_BG)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec*2.0 - 1.0   # [0,1]->[-1,1]
        log["gts"] = gt*2.0 - 1.0   # [0,1]->[-1,1]
        return log


class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer
