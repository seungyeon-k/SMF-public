import time
import os
import torch
import numpy as np
from metrics import averageMeter
from loss import get_loss
import pandas as pd

class TeacherTrainer:
    """Trainer for a conventional iterative training of model"""
    def __init__(self, training_cfg, device):
        self.cfg = training_cfg
        self.device = device
        self.d_val_result = {}
        self.loss = get_loss(training_cfg['loss'], device=device)

    def train(self, model, opt, d_dataloaders, student_loaders=None, logger=None, logdir='', *args, **kwargs):
        cfg = self.cfg
        best_val_loss = np.inf
        best_classify_acc = 0
        time_meter = averageMeter()
        i = 0
        train_loader = d_dataloaders['training']

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
    
            ## teaching
            df = pd.DataFrame()
            for key, item in student_loaders.items():
                student_train_loader = item['training']
                student_test_loader = item['test']
                student_dict = model.student_evaluation_step(
                    student_train_loader, student_test_loader, self.device)
                student_dict['dataset'] = key

                df = df.append(
                    student_dict, ignore_index=True
                )

                savedir = os.path.join(logdir, 'teaching_results')
                if not os.path.exists(savedir):
                    os.makedirs(savedir)

                df.to_csv(os.path.join(savedir, f'df_teaching_results_epoch_{i_epoch}.csv'))

            if (i_epoch+1) % 100 == 0:
                self.save_model(model, logdir, best=False, i_epoch=i_epoch+1)

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