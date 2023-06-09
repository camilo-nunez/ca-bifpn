from .bifpn.model import BiFPN

def builder_bifpn(bifpn_config, backbone_config):
    
    return [BiFPN(bifpn_config.NUM_CHANNELS, 
                  backbone_config.IN_CHANNELS, 
                  first_time=True if _ == 0 else False)
            for _ in range(bifpn_config.NUM_LAYERS)]