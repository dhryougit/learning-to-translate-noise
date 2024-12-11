import argparse
import logging
import math
import os
import random
import sys
import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import to_pil_image

import options as option
from models import create_model

import utils as util
from data import (
    create_dataloader,
    create_dataset,
    create_multi_dataset,
    create_multi_dataloader,
)
from data.data_sampler import DistIterSampler


def init_dist(backend="nccl", **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def mixup_data(x, y, alpha=0.4, use_cuda=True):
    dist = torch.distributions.beta.Beta(torch.tensor([0.4]), torch.tensor([0.4]))
    lam = dist.rsample((1, 1)).item()
    r_index = torch.randperm(y.size(0))
    mixed_y = lam * y + (1 - lam) * y[r_index, :]
    mixed_x = lam * x + (1 - lam) * x[r_index, :]
    return mixed_x, mixed_y


# specific noise level
def add_gaussian_level(images, level):
    batch_size = images.shape[0]
    stds = (
        torch.tensor([level for _ in range(batch_size)])
        .float()
        .view(batch_size, 1, 1, 1)
    )
    noise = torch.randn_like(images) * stds / 255.0
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0, 1)


# 0~level noise
def add_gaussian_randomly(images, level):
    batch_size = images.shape[0]
    stds = torch.rand(batch_size, 1, 1, 1) * level
    noise = torch.randn_like(images) * stds / 255.0
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YAML file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument(
        "--wass_weight",
        type=float,
        required=False,
        default=0.05,
        help="wassertein loss weight",
    )
    parser.add_argument(
        "--spatial_freq_weight",
        type=float,
        required=False,
        default=0.002,
        help="spatial and frequency wassertein weight",
    )
    parser.add_argument(
        "--name", type=str, required=False, default="multitask", help="experiment name"
    )
    parser.add_argument(
        "--seed", type=int, required=False, default=0, help="experiment seed"
    )

    parser.add_argument(
        "--noise_injection_level",
        type=int,
        required=False,
        default=0,
        help="NTNet",
    )

    args = parser.parse_args()
    opt = option.parse(args.opt, args.name, is_train=True)
    opt = option.dict_to_nonedict(opt)

    if opt["trans"] == True : 
        opt["train"]["wass_weight"] = args.wass_weight
        opt["train"]["spatial_freq_weight"] = args.spatial_freq_weight
        opt["network_G_trans"]["setting"]["noise_injection_level"] = args.noise_injection_level

    opt["train"]["manual_seed"] = args.seed
    seed = opt["train"]["manual_seed"]

    util.set_random_seed(seed)

    if args.launcher == "none":
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    if opt["path"].get("resume_state", None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])
    else:
        resume_state = None

    if rank <= 0:
        if resume_state is None:
            util.mkdir_and_rename(opt["path"]["experiments_root"])
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
        if opt["use_wandb_logger"] and "debug" not in opt["name"]:
            import wandb

            project = opt["wandb"]["project"]
            resume_id = opt["logger"].get("resume_id")
            if resume_id:
                wandb_id = resume_id
                resume = "allow"
                logger.warning(f"Resume wandb logger with id={wandb_id}.")
            else:
                wandb_id = wandb.util.generate_id()
                resume = "never"
            wandb.init(
                id=wandb_id,
                resume=resume,
                name=opt["name"],
                config=opt,
                project=project,
                sync_tensorboard=False,
                dir="/131_data/dhryou/wandb",
            )
            wandb.config.update(opt)
            logger.info(f"Use wandb logger with id={wandb_id}; project={project}.")
        else:
            util.setup_logger(
                "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
            )
            logger = logging.getLogger("base")

    if rank <= 0:
        print(opt)
    dataset_ratio = 200
    val_loader_dict = {}
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_multi_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Number of datasets: {:,d}, total images: {:,d}".format(
                        len(opt["datasets"]["train"]["modes"]), len(train_set)
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif "val" in phase:
            val_set = create_dataset(dataset_opt)
            val_sampler = DistributedSampler(
                val_set, num_replicas=world_size, rank=rank, shuffle=False
            )
            val_loader = create_dataloader(val_set, dataset_opt, opt, val_sampler)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
            val_loader_dict[dataset_opt["name"]] = val_loader
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader_dict is not None

    model = create_model(opt)
    device = model.device

    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )
        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)
    else:
        current_step = 0
        start_epoch = 0


    if rank <= 0:
        logger.info(
            "Start training from epoch: {:d}, iter: {:d}".format(
                start_epoch, current_step
            )
        )

    best_psnr = 0.0
    best_iter = 0
    error = mp.Value("b", False)

    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            LQ, GT = train_data["LQ"], train_data["GT"]

            if opt["trans"]:
                LQ = add_gaussian_randomly(LQ, opt["training_noise_level"])
            else:
                LQ = add_gaussian_level(LQ, opt["training_noise_level"])

            if opt["train"]["use_mixup"]:
                LQ, GT = mixup_data(LQ, GT)

            model.feed_data(LQ, GT)
            model.optimize_parameters(current_step)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    if opt["use_wandb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            wandb.log({f"losses/{k}": v}, step=current_step)
                if rank <= 0:
                    logger.info(message)
                    print(message)

            if current_step % opt["train"]["image_log_freq"] == 0 and rank <= 0:
                model.test()
                visuals = model.get_current_total_visuals()
                image_input_array = []
                image_output_array = []
                image_gt_array = []
                for i in range(visuals["Input"].size(0)):
                    image_input_array.append(to_pil_image(visuals["Input"][i]))
                    image_output_array.append(to_pil_image(visuals["Output"][i]))
                    image_gt_array.append(to_pil_image(visuals["GT"][i]))
                wandb.log(
                    {
                        "images/Input": [
                            wandb.Image(image) for image in image_input_array
                        ]
                    },
                    step=current_step,
                )
                wandb.log(
                    {
                        "images/Output": [
                            wandb.Image(image) for image in image_output_array
                        ]
                    },
                    step=current_step,
                )
                wandb.log(
                    {"images/GT": [wandb.Image(image) for image in image_gt_array]},
                    step=current_step,
                )

            if current_step % opt["train"]["val_freq"] == 0:
                ood_avg_psnr_per_level = [0] * len(opt["test_noise_level"])
                ood_avg_ssim_per_level = [0] * len(opt["test_noise_level"])
                total_avg_psnr_per_level = [0] * len(opt["test_noise_level"])
                total_avg_ssim_per_level = [0] * len(opt["test_noise_level"])
                ood_val_dataset_num = 0
                total_val_dataset_num = 0

                metric_values = {}
                for name, val_loader in val_loader_dict.items():
                    level_num = 0
                    for level in opt["test_noise_level"]:
                        avg_psnr = 0.0
                        avg_ssim = 0.0
                        idx = 0
                        for _, val_data in enumerate(val_loader):
                            LQ, GT = val_data["LQ"], val_data["GT"]
                            LQ = add_gaussian_level(LQ, level)

                            model.feed_data(LQ, GT)
                            model.test()
                            visuals = model.get_current_visuals()
                            output = util.tensor2img(visuals["Output"].squeeze())
                            gt_img = util.tensor2img(visuals["GT"].squeeze())
                            avg_psnr += util.calculate_psnr(output, gt_img)
                            avg_ssim += util.calculate_ssim(output, gt_img)
                            idx += 1

                        avg_psnr /= idx
                        avg_psnr_tensor = torch.tensor(avg_psnr).to(device)
                        dist.reduce(avg_psnr_tensor, 0, op=dist.ReduceOp.SUM)
                        avg_psnr = avg_psnr_tensor.item() / world_size

                        avg_ssim /= idx
                        avg_ssim_tensor = torch.tensor(avg_ssim).to(device)
                        dist.reduce(avg_ssim_tensor, 0, op=dist.ReduceOp.SUM)
                        avg_ssim = avg_ssim_tensor.item() / world_size

                        if avg_psnr > best_psnr:
                            best_psnr = avg_psnr
                            best_iter = current_step
                        if rank == 0:
                            logger.info(
                                "# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(
                                    avg_psnr, best_psnr, best_iter
                                )
                            )
                            logger_val = logging.getLogger("val")
                            logger_val.info(
                                "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}, ssim: {:.6f}".format(
                                    epoch, current_step, avg_psnr, avg_ssim
                                )
                            )
                            metric_values[f"metrics/{name}_psnr_{level}"] = avg_psnr
                            metric_values[f"metrics/{name}_ssim_{level}"] = avg_ssim
                            print(
                                f"metrics/{name}_psnr_{level} : {avg_psnr}, {name}_ssim_{level} : {avg_ssim} "
                            )
                        if name != "Sidd" and name != "BSD68_guassian":
                            ood_avg_psnr_per_level[level_num] += avg_psnr
                            ood_avg_ssim_per_level[level_num] += avg_ssim

                        total_avg_psnr_per_level[level_num] += avg_psnr
                        total_avg_ssim_per_level[level_num] += avg_ssim
                        level_num += 1

                    if name != "Sidd" and name != "BSD68_guassian":
                        ood_val_dataset_num += 1
                    total_val_dataset_num += 1

                if rank == 0:
                    for level_num, avg_psnr in enumerate(ood_avg_psnr_per_level):
                        avg_psnr /= ood_val_dataset_num
                        metric_values[
                            f'metrics/OOD_avg_psnr_{opt["test_noise_level"][level_num]}'
                        ] = avg_psnr

                        print(
                            f'metrics/OOD_avg_psnr_{opt["test_noise_level"][level_num]}: {avg_psnr}'
                        )

                    for level_num, avg_ssim in enumerate(ood_avg_ssim_per_level):
                        avg_ssim /= ood_val_dataset_num
                        metric_values[
                            f'metrics/OOD_avg_ssim_{opt["test_noise_level"][level_num]}'
                        ] = avg_ssim
                        print(
                            f'metrics/OOD_avg_ssim_{opt["test_noise_level"][level_num]}: {avg_ssim}'
                        )

                    for level_num, avg_psnr in enumerate(total_avg_psnr_per_level):
                        avg_psnr /= total_val_dataset_num
                        metric_values[
                            f'metrics/total_avg_psnr_{opt["test_noise_level"][level_num]}'
                        ] = avg_psnr
                        print(
                            f'metrics/total_avg_psnr_{opt["test_noise_level"][level_num]}: {avg_psnr}'
                        )

                    for level_num, avg_ssim in enumerate(total_avg_ssim_per_level):
                        avg_ssim /= total_val_dataset_num
                        metric_values[
                            f'metrics/total_avg_ssim_{opt["test_noise_level"][level_num]}'
                        ] = avg_ssim
                        print(
                            f'metrics/total_avg_ssim_{opt["test_noise_level"][level_num]}: {avg_ssim}'
                        )

                    wandb.log(metric_values, step=current_step, commit=True)
            if error.value:
                sys.exit(0)
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank == 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)

    if rank == 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
        wandb.finish()


if __name__ == "__main__":
    main()
