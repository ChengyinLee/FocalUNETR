def get_model(model_name='focalunetr'):
    if model_name == 'focalunetr':
        from networks.focalunetr import FocalUNETR
        model = FocalUNETR(
        img_size=224,
        in_channels=1, 
        out_channels=1, 
        patch_size=2, 
        feature_size=48, 
        depths=[2,2,6,2], 
        focal_levels=[2,2,2,2], 
        expand_sizes=[3,3,3,3], 
        expand_layer='all', 
        num_heads=[4,8,16,32], 
        focal_windows=[7,5,3,1], 
        window_size=7,
        use_conv_embed=True, 
        use_shift=False)
    elif model_name == 'focalunetr_contour':
        from networks.focalunetr_with_contour  import FocalUNETR
        model = FocalUNETR(
        img_size=224,
        in_channels=1, 
        out_channels=1, 
        patch_size=2, 
        feature_size=48, 
        depths=[2,2,6,2], 
        focal_levels=[2,2,2,2], 
        expand_sizes=[3,3,3,3], 
        expand_layer='all', 
        num_heads=[4,8,16,32], 
        focal_windows=[7,5,3,1], 
        window_size=7, 
        use_conv_embed=True, 
        use_shift=False)
    else:
        model = None
        print('Please use a correct model name!')
    return model