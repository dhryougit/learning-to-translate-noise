"""create dataset and dataloader"""

import logging

import torch
import torch.utils.data
import numpy as np
import random


g = torch.Generator()
g.manual_seed(0)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_multi_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt["phase"]
    if phase == "train":
        if opt["dist"]:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt["n_workers"]
            assert dataset_opt["batch_size"] % world_size == 0
            batch_size = dataset_opt["batch_size"] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt["n_workers"] * len(opt["gpu_ids"])
            batch_size = dataset_opt["batch_size"]
            shuffle = True
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=(phase == "val"),
            worker_init_fn=seed_worker,
            generator=g,
        )


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt["phase"]
    if phase == "train":
        if opt["dist"]:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt["n_workers"]
            assert dataset_opt["batch_size"] % world_size == 0
            batch_size = dataset_opt["batch_size"] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt["n_workers"] * len(opt["gpu_ids"])
            batch_size = dataset_opt["batch_size"]
            shuffle = True
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=(phase == "val"),
            worker_init_fn=seed_worker,
            generator=g,
        )


def create_dataset(dataset_opt):
    mode = dataset_opt["mode"]

  
    from data.LQGT_dataset import LQGTDataset as D

    dataset = D(dataset_opt, mode)
    # raise NotImplementedError("Datasetx [{:s}] is not recognized.".format(mode))

    logger = logging.getLogger("base")
    logger.info(
        "Dataset [{:s} - {:s}] is created.".format(
            dataset.__class__.__name__, dataset_opt["name"]
        )
    )
    return dataset


def create_multi_dataset(dataset_opt):
    set_of_dataset = []
    for mode in dataset_opt["modes"]:


        from data.LQGT_dataset import LQGTDataset as D
        dataset = D(dataset_opt, mode)
        # raise NotImplementedError("Dataset [{:s}] is not recognized.".format(mode))

        logger = logging.getLogger("base")
        logger.info(
            "Dataset [{:s} - {:s}] is created.".format(
                dataset.__class__.__name__, dataset_opt["name"]
            )
        )
        set_of_dataset.append(dataset)

        from data.Mixed_dataset import MixedDataset as MD

        dataset = MD(dataset_opt, set_of_dataset)
    return dataset
