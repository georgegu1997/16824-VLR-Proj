def getDataloaders(cfg):
    if cfg.dataset.name == "rps":
        from .robustpointset import getDataloaders as getDls
        return getDls(cfg)
    else:
        raise Exception("Unknown cfg.dataset.name =", cfg.dataset.name)