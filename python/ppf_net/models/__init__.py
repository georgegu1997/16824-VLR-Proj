from .pointnet import PointNetCls
from .pointnet2 import PointNet2Cls
from .dgcnn import DGCNNCls

def getModel(cfg):

    if cfg.model.name == "pn":
        ModelClass = PointNetCls
    elif cfg.model.name == "pn2":
        ModelClass = PointNet2Cls
    elif cfg.model.name == "dgcnn":
        ModelClass = DGCNNCls
    else:
        raise Exception("Unknown cfg.model.name =", cfg.model.name)

    if cfg.weights_path is None:
        model = ModelClass(cfg)
    else:
        model = ModelClass.load_from_checkpoint(cfg.weights_path, config=cfg)
    
    return model