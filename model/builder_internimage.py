from .internimage.intern_image import InternImage

from torch.hub import load_state_dict_from_url

def builder_im(config, out_indices=(0, 1, 2, 3)):
    model = InternImage(norm_layer='LN',
                        drop_path_rate=config.DROP_PATH_RATE,
                        layer_scale=config.LAYER_SCALE,
                        post_norm=config.POST_NORM,
                        with_cp=config.WITH_CP,
                        core_op=config.CORE_OP,
                        channels=config.CHANNELS,
                        depths=config.DEPTHS,
                        groups=config.GROUPS,
                        mlp_ratio=config.MLP_RATIO,
                        offset_scale=config.OFFSET_SCALE,
                        out_indices=out_indices,
                       )
    checkpoint = load_state_dict_from_url(url=config.URL, map_location="cpu")
    out_n = model.load_state_dict(checkpoint['model'], strict=False)
    
    if len(out_n.unexpected_keys)!=0: print(f'[+] The unexpected keys was: {out_n.unexpected_keys}')
    
    for param in model.parameters():
        param.requires_grad = False

    return model