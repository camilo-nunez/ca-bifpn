from omegaconf import OmegaConf

def default_config():
    # Main Config
    _C = OmegaConf.create()

    ## Model config
    _C.MODEL = OmegaConf.create() 

    _C.MODEL.BACKBONE = OmegaConf.create()
    _C.MODEL.BACKBONE.NAME = str()
    _C.MODEL.BACKBONE.CORE_OP = 'DCNv3'
    _C.MODEL.BACKBONE.CHANNELS = int()
    _C.MODEL.BACKBONE.DEPTHS = list()
    _C.MODEL.BACKBONE.GROUPS = list()
    _C.MODEL.BACKBONE.MLP_RATIO = float()
    _C.MODEL.BACKBONE.OFFSET_SCALE = float()
    _C.MODEL.BACKBONE.POST_NORM = False
    _C.MODEL.BACKBONE.DROP_PATH_RATE = 0.4
    _C.MODEL.BACKBONE.LAYER_SCALE = 1.0
    _C.MODEL.BACKBONE.WITH_CP = False
    _C.MODEL.BACKBONE.IN_CHANNELS = list()
    _C.MODEL.BACKBONE.IMAGE_SIZE = int()
    _C.MODEL.BACKBONE.URL = str()

    _C.MODEL.BIFPN = OmegaConf.create()
    _C.MODEL.BIFPN.TYPE = str()
    _C.MODEL.BIFPN.NAME = str()
    _C.MODEL.BIFPN.NUM_CHANNELS = int()
    _C.MODEL.BIFPN.NUM_LAYERS = int()
    
    ## Dataset config
    _C.DATASET = OmegaConf.create()
    
    _C.DATASET.NAME = str()
    _C.DATASET.PATH = str()
    _C.DATASET.TRAIN_SET = str()
    _C.DATASET.VAL_SET = str()
    _C.DATASET.MEAN = list()
    _C.DATASET.STD = list()
    
    _C.DATASET.OBJ_LIST = list()
    
    ## Train Config
    _C.TRAIN = OmegaConf.create()
    _C.TRAIN.BASE_LR = int()
    _C.TRAIN.OPTIM = str()
    _C.TRAIN.NUM_EPOCHS = int()
    _C.TRAIN.VAL_INTERVAL = int()
#     _C.TRAIN.WEIGHT_DECAY = 0.05
    _C.TRAIN.USE_CHECKPOINT = False
    _C.TRAIN.CHECKPOINT_PATH = str()
    _C.TRAIN.BATCH_SIZE = int()

    return _C

def create_train_config(args):
    
    model_conf = OmegaConf.load(args.cfg_model)
    dataset_conf = OmegaConf.load(args.cfg_dataset)
    def_config = default_config()

    base_config = OmegaConf.merge(def_config, model_conf, dataset_conf)
    
    if hasattr(args, 'fpn_type') and args.fpn_type:
        base_config.MODEL.BIFPN.TYPE = args.fpn_type
    
    if hasattr(args, 'lr') and args.lr:
        base_config.TRAIN.BASE_LR = args.lr
    if hasattr(args, 'optim') and args.optim:
        base_config.TRAIN.OPTIM = args.optim
    if hasattr(args, 'num_epochs') and args.num_epochs:
        base_config.TRAIN.NUM_EPOCHS = args.num_epochs
    if hasattr(args, 'val_interval') and args.val_interval:
        base_config.TRAIN.VAL_INTERVAL = args.val_interval
    if hasattr(args, 'batch_size') and args.batch_size:
        base_config.TRAIN.BATCH_SIZE = args.batch_size
        
    if hasattr(args, 'dataset_path') and args.dataset_path:
        base_config.DATASET.PATH = args.dataset_path

    if hasattr(args, 'use_checkpoint') and args.use_checkpoint:
        base_config.TRAIN.USE_CHECKPOINT = args.use_checkpoint
        if hasattr(args, 'path_checkpoint') and args.path_checkpoint:
            base_config.TRAIN.CHECKPOINT_PATH = args.path_checkpoint
        else: raise RuntimeError('You must specify the \'--path_checkpoint\'. ')

    return base_config