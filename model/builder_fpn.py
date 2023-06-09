from .bifpn.model import BiFPN
from .cabifpn.model import CABiFPN

def builder_fpn(base_config):
    if base_config.MODEL.BIFPN.TYPE == 'bs':
        fpn_buider = BiFPN
        print('[+] BiFPN loaded')
    elif base_config.MODEL.BIFPN.TYPE == 'ca':
        fpn_buider = CABiFPN
        print('[+] CABiFPN loaded')
    else:
        raise RuntimeError("[+] FPN backbone does not exist !. Select the FPN backbone, this should be \'bs\' (refer to baseline) or \'ca\' (refer to context agregation.")
            
    return [fpn_buider(base_config.MODEL.BIFPN.NUM_CHANNELS,
                       base_config.MODEL.BACKBONE.IN_CHANNELS, 
                       first_time=True if _ == 0 else False)
            for _ in range(base_config.MODEL.BIFPN.NUM_LAYERS)]