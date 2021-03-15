import torch
from torch import nn, optim 
from torch.utils import data 
from torchnet import meter 
from tqdm import tqdm 
import numpy as np 
import os 
from datetime import datetime 
import traceback 

from loggers import TensorboardLogger
from utils.device import move_to, detach 

class Trainer():
    def __init__(self, 
                device, 
                config,
                model, 
                criterion,
                optimizer,
                scheduler,
                metric
                ):
        super(Trainer, self).__init__() 

        self.config = config 
        self.device = device
        self.model = model 
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = metric 

        self.train_ID = self.config.get('id', None)
        self.train_ID += '-' + datetime.now().strftime('%Y_%m_%d-%H_%M_%S') 

        self.nepochs = self.config['trainer']['nepochs']
        self.log_step = self.config['trainer']['log_step']
        self.val_step = self.config['trainer']['val_step']
        
        self.best_loss = np.inf 
        self.best_metric = {k: 0.0 for k in self.metric.keys()}
        self.val_loss = list() 
        self.val_metric = {k: list() for k in self.metric.keys() }

        self.save_dir = os.path.join(config['trainer']['save_dir'], self.train_ID)
        self.tsboard = TensorboardLogger(path=self.save_dir) 

    def save_checkpoint(self, epoch, val_loss = 0, val_metric = None):
        data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }

        if val_loss < self.best_loss:
            print(
                f'Loss is improved from {self.best_loss: .6f} to {val_loss: .6f}. Saving weights...')
            torch.save(data, os.path.join(self.save_dir, 'best_loss.pth'))
            # Update best_loss
            self.best_loss = val_loss
        else:
            print(f'Loss is not improved from {self.best_loss:.6f}.')

        # for k in self.metric.keys():
        #    if val_metric[k] > self.best_metric[k]:
        #        print(
        #            f'{k} is improved from {self.best_metric[k]: .6f} to {val_metric[k]: .6f}. Saving weights...')
        #        torch.save(data, os.path.join(
        #            self.save_dir, f'best_metric_{k}.pth'))
        #        self.best_metric[k] = val_metric[k]
        #    else:
        #        print(
        #            f'{k} is not improved from {self.best_metric[k]:.6f}.')

        print('Saving current model...')
        torch.save(data, os.path.join(self.save_dir, 'current.pth'))    

    def save_current_checkpoint(self, epoch):
        data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        print('Saving mini current model...')
        torch.save(data, os.path.join(self.save_dir, 'mini_current.pth'))    
    
    def train_epoch(self, epoch, dataloader):
        total_loss = meter.AverageValueMeter() 
        loss1 = meter.AverageValueMeter() 
        loss2 = meter.AverageValueMeter() 

        self.model.train()
        print("Training..........")
        progress_bar = tqdm(dataloader)
        max_iter = len(dataloader)
        for i, data in enumerate(progress_bar):
            Fs, Ms, num_objects, info = data 
            
            Fs = move_to(Fs, self.device)
            Ms = move_to(Ms, self.device)
            num_objects = move_to(torch.tensor([num_objects]), self.device) 
            
            self.optimizer.zero_grad() 
            
            # memorize
            key_0, value_0 = self.model(Fs[:,:,0], Ms[:,:,0], num_objects) 

            logit_1 = self.model(Fs[:,:,1], key_0, value_0, num_objects)

            loss_1 = self.criterion(logit_1, Ms[:,:,1])

            key_1, value_1 = self.model(Fs[:,:,1], Ms[:,:,1], num_objects) 
            
            this_keys = torch.cat([key_0, key_1], dim=3)
            this_values = torch.cat([value_0, value_1], dim=3)
        
            logit_2 = self.model(Fs[:,:,2], this_keys, this_values, num_objects)
             
            loss_2 = self.criterion(logit_2, Ms[:,:,2])

            loss = loss_1 * 2 + loss_2
            
            if loss == 0 or not torch.isfinite(loss):
                    continue

            loss1.add(loss_1.item()) 
            loss2.add(loss_2.item()) 
            total_loss.add(loss.item()) 

            loss.backward() 

            self.optimizer.step() 

            with torch.no_grad():
                total_loss.add(loss.item())
                progress_bar.set_description(
                    'Iteration: {}/{}. Loss 1 - 1: {:.5f}. Loss 2 - 1: {:.5f}. Total loss: {:.5f}'.format(
                        i + 1, len(dataloader), loss_1.item(),
                        loss_2.item(), total_loss.value()[0]))
                
                self.tsboard.update_scalar(
                    'Train_Loss/total', loss, epoch * len(dataloader) + i 
                )
                self.tsboard.update_scalar(
                    'Train_Loss/1-1', loss_1, epoch * len(dataloader) + i 
                )
                self.tsboard.update_scalar(
                    'Train_Loss/2-1', loss_2, epoch * len(dataloader) + i 
                )

            # if (i + 1) % self.config['trainer']['checkpoint_mini_step'] == 0:
            #     self.save_current_checkpoint(epoch)
                
        print("+ Train result")
        avg_loss = total_loss.value()[0]
        print("Loss:", avg_loss)
        # for m in self.metric.values():
        #     m.summary() 

    @torch.no_grad() 
    def val_epoch(self, epoch, dataloader):
        total_loss = meter.AverageValueMeter() 
        loss1 = meter.AverageValueMeter() 
        loss2 = meter.AverageValueMeter() 

        self.model.eval() 
        print("Evaluating.....")
        progress_bar = tqdm(dataloader)
        # cls_loss
        for i, data in enumerate(progress_bar):
            Fs, Ms, num_objects, info = data 
            
            Fs = move_to(Fs, self.device)
            Ms = move_to(Ms, self.device)
            num_objects = move_to(torch.tensor([num_objects]), self.device) 
           
            self.optimizer.zero_grad() 
            # memorize
            key_0, value_0 = self.model(Fs[:,:,0], Ms[:,:,0], num_objects) 

            logit_1 = self.model(Fs[:,:,1], key_0, key_1, num_objects)

            loss_1 = self.criterion(logit_1, Ms[:,:,1])

            key_1, value_1 = self.model(Fs[:,:,1], Ms[:,:,1], num_objects) 
            
            this_keys = torch.cat([key_0, key_1], dim=3)
            this_values = torch.cat([value_0, value_1], dim=3)
        
            logit_2 = self.model(Fs[:,:,2], this_keys, this_values, num_objects)
             
            loss_2 = self.criterion(logit_2, Ms[:,:,2])

            loss = loss_1 * 2 + loss_2
            
            if loss == 0 or not torch.isfinite(loss):
                    continue

            loss1.add(loss_1.item()) 
            loss2.add(loss_2.item()) 
            total_loss.add(loss.item()) 
            
            total_loss.add(loss.item())
            progress_bar.set_description(
                'Iteration: {}/{}. Loss 1 - 1: {:.5f}. Loss 2 - 1: {:.5f}. Total loss: {:.5f}'.format(
                    i + 1, len(dataloader), loss_1.item(),
                    loss_2.item(), total_loss.value()[0]))
            
            
        print("+ Evaluation result")
        avg_loss = total_loss.value()[0]
        print("Loss: ", avg_loss)
        
        self.val_loss.append(avg_loss)
        
        self.tsboard.update_scalar(
            'Val_Loss/total', total_loss.value()[0], epoch * len(dataloader) + i 
        )
        self.tsboard.update_scalar(
            'Val_Loss/1-1', loss1.value()[0], epoch * len(dataloader) + i 
        )
        self.tsboard.update_scalar(
            'Val_Loss/2-1', loss2.value()[0], epoch * len(dataloader) + i 
        )
                        
        # Calculate metric here
        # for k in self.metric.keys():
        #     m = self.metric[k].value()
        #     self.metric[k].summary()
        #     self.val_metric[k].append(m)
        #     self.tsboard.update_metric('val', k, m, epoch)

    def train(self, train_dataloader, val_dataloader):
        for epoch in range(self.nepochs):
            print('\nEpoch {:>3d}'.format(epoch))
            print('-----------------------------------')

            # Note learning rate
            for i, group in enumerate(self.optimizer.param_groups):
                self.tsboard.update_lr(i, group['lr'], epoch)

            # 1: Training phase
            self.train_epoch(epoch=epoch, dataloader=train_dataloader)

            print()

            # 2: Evalutation phase
            if (epoch + 1) % self.val_step == 0:
                # 2: Evaluating model
                self.val_epoch(epoch, dataloader=val_dataloader)
                print('-----------------------------------')

                # 3: Learning rate scheduling
                self.scheduler.step(self.val_loss[-1])

                # 4: Saving checkpoints

                # if not self.debug:
                # Get latest val loss here
                val_loss = self.val_loss[-1]
                val_metric = None 
                # {k: m[-1] for k, m in self.val_metric.items()}
                self.save_checkpoint(epoch, val_loss, val_metric)

            # self.save_checkpoint(epoch)
            
               