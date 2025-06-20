import os
import time
import torch
import sys

sys.path.append('..')

import lib.models as models
import lib.networks as networks
import wandb
from .base_trainer import BaseTrainer
from lib.data.med_transforms import get_mae_pretrain_transforms
from lib.data.med_datasets import get_train_loader
from torchvision import transforms
import numpy as np


class MAE3DTrainer(BaseTrainer):
    """
    3D Masked Autoencoder Trainer
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'MAE3D'
        self.toPIL = transforms.ToPILImage()

    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name} of arch {args.arch}")
            self.model = getattr(models, self.model_name)(
                encoder=getattr(networks, args.enc_arch),
                decoder=getattr(networks, args.dec_arch),
                args=args)
            self.wrap_model()
        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")

    def build_optimizer(self):
        assert (self.model is not None and self.wrapped_model is not None), \
            "Model is not created and wrapped yet. Please create model first."
        print("=> creating optimizer")
        args = self.args
        optim_params = self.get_parameter_groups()
        self.optimizer = torch.optim.AdamW(optim_params,
                                           lr=args.lr,
                                           betas=(args.beta1, args.beta2),
                                           weight_decay=args.weight_decay)

    def build_dataloader(self):
        if self.dataloader is None:
            print("=> creating dataloader")
            args = self.args
            train_transform = get_mae_pretrain_transforms(args)
            self.dataloader = get_train_loader(args,
                                               batch_size=self.batch_size,
                                               workers=self.workers,
                                               train_transform=train_transform)
            self.iters_per_epoch = len(self.dataloader)
            print(f"==> Length of train dataloader is {self.iters_per_epoch}")
        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")
        print("=> finish creating dataloader")

    def run(self):
        args = self.args
        # Compute iterations when resuming
        niters = args.start_epoch * self.iters_per_epoch

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                self.dataloader.sampler.set_epoch(epoch)
                torch.distributed.barrier()

            # train for one epoch
            niters = self.epoch_train(epoch, niters)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                if epoch == 0 or (epoch + 1) % args.save_freq == 0:
                    print(f"=> start saving checkpoint after epoch {epoch + 1}")
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    }, is_best=False, filename=f'{args.ckpt_dir}/checkpoint_{epoch:04d}.pth.tar')
                    print("=> finish saving checkpoint")

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.wrapped_model
        optimizer = self.optimizer

        grid_size = args.grid_size
        num_patches = int(np.prod(grid_size))
        smooth = torch.eye(num_patches)  # one hot

        model.train()

        load_start_time = time.time()
        for i, batch_data in enumerate(train_loader):
            load_time = time.time() - load_start_time
            # Adjust learning at the beginning of each iteration
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)
            # For SSL pretraining, only image data is required for training
            image = batch_data['image']
            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)
            # Compute output and loss
            forward_start_time = time.time()
            loss = model(image, smooth, return_image=False)
            forward_time = time.time() - forward_start_time
            # Compute gradient and do SGD step
            bp_start_time = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bp_time = time.time() - bp_start_time
            # Log to the screen
            if i % args.print_freq == 0:
                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.iters_per_epoch} | "
                      f"TotalIter: {niters:06d} | "
                      f"Init Lr: {self.lr:.05f} | "
                      f"Lr: {optimizer.param_groups[0]['lr']:.05f} | "
                      f"Load Time: {load_time:.03f}s | "
                      f"Forward Time: {forward_time:.03f}s | "
                      f"Backward Time: {bp_time:.03f}s | "
                      f"Loss: {loss.item():.03f}")
                if args.rank == 0:
                    wandb.log(
                        {
                            "lr": optimizer.param_groups[0]['lr'],
                            "Loss": loss.item(),
                        },
                        step=niters,
                    )
            niters += 1
            load_start_time = time.time()
        return niters

    def resume(self):
        args = self.args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
