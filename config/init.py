from typing import List, Tuple
from dataclasses import dataclass, MISSING
from omegaconf import OmegaConf

import os

@dataclass
class BACKBONE:
    MODEL_NAME: str = MISSING
    OUT_INDICES: List[int] = MISSING
        
@dataclass
class NECK:
    MODEL_NAME: str = MISSING
    IN_CHANNELS: List[int] = MISSING
    NUM_CHANNELS: int = MISSING
    NUM_LAYERS: int = MISSING

@dataclass
class DATASET:
    NAME: str = MISSING
    PATH: str = MISSING
    TRAIN_SET: str = MISSING
    VAL_SET: str = MISSING
    MEAN: List[float] = MISSING
    STD: List[float] = MISSING
    OBJ_LIST: List[str] = MISSING
    IMAGE_SIZE: int = MISSING
        
@dataclass
class ENVTRAIN:
    NUM_EPOCHS: int = MISSING
    BATCH_SIZE: int = MISSING
    CHECKPOINT_PATH: str = MISSING
    CHECKPOINT_USE: bool = MISSING
    CHECKPOINT_FN: str = MISSING
        
@dataclass
class ENVTRAIN:
    NUM_EPOCHS: int = MISSING
    BATCH_SIZE: int = MISSING
    CHECKPOINT_PATH: str = MISSING
    CHECKPOINT_USE: bool = MISSING
    CHECKPOINT_FN: str = MISSING

@dataclass
class OPTIM:
    BASE_LR: float = MISSING
    WEIGHT_DECAY: float = MISSING
    MOMENTUM: float = MISSING
        
@dataclass
class SCHEDULER:
    T_0: int = MISSING
    T_MULT: float = MISSING

def default_config():
    # Main Config
    _C = OmegaConf.create()

    ## Model config
    _C.MODEL = OmegaConf.create()
    _C.MODEL.BACKBONE = OmegaConf.structured(BACKBONE)
    _C.MODEL.NECK = OmegaConf.structured(NECK)
    
    ## Dataset config
    _C.DATASET = OmegaConf.structured(DATASET)
    
    ## Train config
    _C.TRAIN = OmegaConf.create()
    ### Traning eviroment variables
    _C.TRAIN.ENV = OmegaConf.structured(ENVTRAIN)
    ### Optimizer
    _C.TRAIN.OPTIM = OmegaConf.structured(OPTIM)
    ### Scheduler
    _C.TRAIN.SCHEDULER = OmegaConf.structured(SCHEDULER)

    return _C

def create_train_config_A(args):
    
    model_conf = OmegaConf.load(args.cfg_model)
    dataset_conf = OmegaConf.load(args.cfg_dataset)
    base_config = default_config()
    
    base_config.MODEL = OmegaConf.merge(base_config.MODEL, model_conf)
    
    base_config.DATASET = OmegaConf.merge(base_config.DATASET, dataset_conf)
    if hasattr(args, 'dataset_path') and args.dataset_path:
        base_config.DATASET.PATH = args.dataset_path
    
    if hasattr(args, 'num_epochs') and args.num_epochs:
        base_config.TRAIN.ENV.NUM_EPOCHS = args.num_epochs
    if hasattr(args, 'batch_size') and args.batch_size:
        base_config.TRAIN.ENV.BATCH_SIZE = args.batch_size
        
    if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
        base_config.TRAIN.ENV.CHECKPOINT_PATH = args.checkpoint_path
    if hasattr(args, 'use_checkpoint') and args.use_checkpoint:
        base_config.TRAIN.ENV.CHECKPOINT_USE = args.use_checkpoint
        if hasattr(args, 'checkpoint_fn') and args.checkpoint_fn:
            base_config.TRAIN.ENV.CHECKPOINT_FN = args.checkpoint_fn
        else: raise RuntimeError('You must specify the \'--checkpoint_fn\'. ')
        
    if hasattr(args, 'lr') and args.lr:
        base_config.TRAIN.OPTIM.BASE_LR = args.lr
    if hasattr(args, 'wd') and args.wd:
        base_config.TRAIN.OPTIM.WEIGHT_DECAY = args.wd

    return base_config