
def getDataloaders(cfg):
    if cfg.dataset.name == None:
        return None
    else:
        raise Exception("Unknown cfg.dataset.name =", cfg.dataset.name)