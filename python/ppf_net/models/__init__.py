from .pointnet import PointNetCls

def getModel(cfg):

    if cfg.model.name == "pn":
        ModelClass = PointNetCls
    else:
        raise Exception("Unknown cfg.model.name =", cfg.model.name)

    if cfg.weights_path is None:
        model = ModelClass(cfg)
    else:
        model = ModelClass.load_from_checkpoint(cfg.weights_path, config=cfg)
    
    return model