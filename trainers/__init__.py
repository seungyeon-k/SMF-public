from trainers.logger import BaseLogger
from trainers.base import BaseTrainer
from trainers.teacher import TeacherTrainer

def get_trainer(cfg):
    trainer_type = cfg.get('trainer', None)
    device = cfg['device']
    if trainer_type == 'base':
        trainer = BaseTrainer(cfg['training'], device=device)
    elif trainer_type == 'teacher':
        trainer = TeacherTrainer(cfg['training'], device=device)
    return trainer

def get_logger(cfg, writer):
    logger_cfg = cfg['logger']
    logger = BaseLogger(writer, **logger_cfg)
    return logger