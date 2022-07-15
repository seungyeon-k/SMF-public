import time
import os
import torch
import numpy as np
from metrics import averageMeter
from loss import get_loss

class BaseTrainer:
    """Trainer for a conventional iterative training of model"""
    def __init__(self, training_cfg, device):
        self.cfg = training_cfg
        self.device = device
        self.d_val_result = {}
        self.loss = get_loss(training_cfg['loss'], device=device)

    def train(self, model, opt, d_dataloaders, logger=None, logdir='', *args, **kwargs):
        cfg = self.cfg
        # show_latent = cfg.get('show_latent', True)
        best_val_loss = np.inf
        best_classify_acc = 0
        time_meter = averageMeter()
        i = 0
        train_loader = d_dataloaders['training']
        do_val = False
        if 'validation' in d_dataloaders.keys():
            val_loader = d_dataloaders['validation']
            do_val = True

        for i_epoch in range(cfg.n_epoch):
            for output in train_loader:
                
                i += 1

                model.train()
                x = output[0].to(self.device)
                
                start_ts = time.time()
                d_train = model.train_step(x, optimizer=opt, loss=self.loss)

                time_meter.update(time.time() - start_ts)
                logger.process_iter_train(d_train)

                if i % cfg.print_interval == 0:
                    d_train = logger.summary_train(i)
                    print(f"Iter [{i:d}] Avg Loss: {d_train['loss/train_loss_']:.4f} Elapsed time: {time_meter.sum:.4f}")
                    time_meter.reset()
                
                if do_val:
                    if i % cfg.val_interval == 0:
                        model.eval()
                        for output in val_loader:
                            val_x = output[0].to(self.device)
                            label = output[1].to(self.device)

                            d_val = model.validation_step(val_x, loss=self.loss, y=label)
                            logger.process_iter_val(d_val)
                        d_val = logger.summary_val(i)
                        val_loss = d_val['loss/val_loss_']
                        print(d_val['print_str'])
                        best_model = val_loss < best_val_loss

                        # if i % cfg.save_interval == 0 or best_model:
                        #     self.save_model(model, logdir, best=best_model, i_iter=i)
                        if best_model:
                            print(f'Iter [{i:d}] best model saved {val_loss} <= {best_val_loss}')
                            best_val_loss = val_loss
                
            if 'classification_interval_epoch' in cfg.keys():
                if (i_epoch+1) % cfg.classification_interval_epoch == 0:
                    tic = time.time()
                    model.eval()
                    classify_dict = model.classification_step(train_loader, val_loader, self.device)
                    classify_acc = classify_dict['classify_acc_']
                    logger.writer.add_scalar('classify_acc_', classify_acc, i)
                    best_classify = best_classify_acc < classify_acc
                    if best_classify:
                        print(f'Epoch [{i_epoch:d}] best classify model saved {best_classify_acc} <= {classify_acc}')
                        best_classify_acc = classify_acc
                        self.save_model(model, logdir, best=False, i_iter=i, best_classify=True)
                    else:
                        print(f'Epoch [{i_epoch:d}] classification acc: {classify_acc}')
                    toc = time.time()
                    print(f'time spent for classification : {toc-tic} s')

            if 'interpolation_interval_epoch' in cfg.keys():
                if (i_epoch+1) % cfg.interpolation_interval_epoch == 0:
                    model.eval()
                    interpolation_dict = model.interpolation_step(train_loader, self.device)

                    logger.writer.add_mesh('interpolation_inter_class(', vertices=interpolation_dict[0].transpose(1,0).unsqueeze(0), global_step=i)
                    if len(interpolation_dict) == 2:
                        logger.writer.add_mesh('interpolation_intra_class(', vertices=interpolation_dict[1].transpose(1,0).unsqueeze(0), global_step=i)

            if i_epoch % cfg.save_interval == 0:
                self.save_model(model, logdir, best=False, i_epoch=i_epoch)
            
        self.save_model(model, logdir, best=False, last=True)
        return model, best_val_loss

    def save_model(self, model, logdir, best=False, last=False, i_iter=None, i_epoch=None, best_classify=False):
        if last:
            pkl_name = "model_last.pkl"
        else:
            if best:
                pkl_name = "model_best.pkl"
            elif best_classify:
                pkl_name = "model_best_classify.pkl"
            else:
                if i_iter is not None:
                    pkl_name = "model_iter_{}.pkl".format(i_iter)
                else:
                    pkl_name = "model_epoch_{}.pkl".format(i_epoch)
        state = {"epoch": i_epoch, "model_state": model.state_dict(), 'iter': i_iter}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
        print(f'Model saved: {pkl_name}')