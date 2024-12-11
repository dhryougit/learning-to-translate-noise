import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm
from ema_pytorch import EMA

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.modules.loss import MatchingLoss
from models.modules.loss import WassLoss

from .base_model import BaseModel
import torch.nn.functional as F


import matplotlib.pyplot as plt


logger = logging.getLogger("base")


class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.model = networks.define_G(opt).to(self.device)
        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )
        else:
            self.model = DataParallel(self.model)

        # define network and load pretrained models
        self.trans = False
        if opt["trans"]:
            self.trans = True
            self.trans_model = networks.define_G_trans(opt).to(self.device)
            if opt["dist"]:
                self.trans_model = DistributedDataParallel(
                    self.trans_model, device_ids=[torch.cuda.current_device()]
                )
            else:
                self.trans_model = DataParallel(self.trans_model)

        print("translation: ", self.trans)
        self.load()

        if self.is_train:
            self.model.train()
            if self.trans == True:
                self.trans_model.train()

            is_weighted = opt["train"]["is_weighted"]
            loss_type = opt["train"]["loss_type"]
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            if self.trans == True:
                self.loss_wass = WassLoss(
                    spatial_freq_weight=opt["train"]["spatial_freq_weight"]
                ).to(self.device)
                self.wass_weight = train_opt["wass_weight"]
            self.weight = opt["train"]["weight"]

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (k,v) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if self.trans == True:
                optim_params_trans = []
                for (k, v ) in ( self.trans_model.named_parameters()  ):  # can optimize for a part of the model
                    if v.requires_grad:
                        optim_params_trans.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt["optimizer"] == "Adam":
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt["optimizer"] == "AdamW":
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )

                if self.trans == True:
                    self.optimizer_trans = torch.optim.AdamW(
                        optim_params_trans,
                        lr=train_opt["lr_G"],
                        weight_decay=wd_G,
                        betas=(train_opt["beta1"], train_opt["beta2"]),
                    )
            elif train_opt["optimizer"] == "Lion":
                self.optimizer = Lion(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print("Not implemented optimizer, default using Adam!")

            self.optimizers.append(self.optimizer)
            if self.trans == True:
                self.optimizers.append(self.optimizer_trans)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"],
                        )
                    )
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            if self.trans == True:
                self.ema2 = EMA(self.trans_model, beta=0.995, update_every=10).to(
                    self.device
                )
            self.log_dict = OrderedDict()

    def feed_data(self, LQ, GT):
        self.lq = LQ.to(self.device)  # LQ
        if GT is not None:
            self.gt = GT.to(self.device)  # GT

    def plot_and_save(self, tensor, title, filename):
        """Plot and save the tensor distribution as a histogram."""
        tensor = tensor.detach().cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.title(title)

        # if tensor.ndim == 4:  # For batch of images (B, C, H, W)
        #     tensor = tensor.mean(axis=(1, 2, 3))  # Average over all dimensions for visualization

        # if tensor.ndim == 3:  # For batch of single-channel images (B, H, W)
        #     tensor = tensor.mean(axis=(1, 2))  # Average over H and W dimensions

        # Plot the histogram
        plt.hist(tensor.flatten(), bins=1000, color="blue")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        # Save the plot as a file
        plt.savefig(filename)
        plt.close()

    def optimize_parameters(self, step):

        self.optimizer.zero_grad()
        if self.trans == True:
            self.optimizer_trans.zero_grad()
            self.trans_input = self.trans_model(self.lq)
            preds = self.model(self.trans_input)
            loss = self.weight * self.loss_fn(preds, self.gt)
            wass_loss = self.loss_wass(self.trans_input, self.gt)

            loss = loss + wass_loss * self.wass_weight
            loss.backward()

            self.optimizer_trans.step()
            self.ema2.update()

        else:
            preds = self.model(self.lq)
            loss = self.weight * self.loss_fn(preds, self.gt)
            loss.backward()
            self.optimizer.step()
            self.ema.update()

        # set log
        self.log_dict["loss"] = loss.item()

    def test(self):

        self.model.eval()
        if self.trans == True:
            self.trans_model.eval()
            with torch.no_grad():
                self.output = self.model(self.trans_model(self.lq))

        else:
            with torch.no_grad():
                self.output = self.model(self.lq)

        self.model.train()
        if self.trans == True:
            self.trans_model.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.lq.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.gt.detach()[0].float().cpu()
        return out_dict

    def get_current_total_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.lq.detach().float().cpu()
        out_dict["Output"] = self.output.detach().float().cpu()
        if need_GT:
            out_dict["GT"] = self.gt.detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])
        if self.trans == True:
            load_path_G_trans = self.opt["path"]["pretrain_model_trans"]
            if load_path_G_trans is not None:
                logger.info(
                    "Loading model for translation [{:s}] ...".format(load_path_G_trans)
                )
                self.load_network(
                    load_path_G_trans, self.trans_model, self.opt["path"]["strict_load"]
                )

    def save(self, iter_label):
        if self.trans == True:
            self.save_network(self.trans_model, "Trans", iter_label)
        else:
            self.save_network(self.model, "G", iter_label)

        self.save_network(self.ema.ema_model, "EMA", "lastest")
