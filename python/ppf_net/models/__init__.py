def getModel(cfg):

    ModelClass = None

    if cfg.weights_path is None:
        model = ModelClass(cfg)
    else:
        model = ModelClass.load_from_checkpoint(cfg.weights_path, config=cfg)
    
    return model