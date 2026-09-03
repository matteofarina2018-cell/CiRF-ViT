"""Model families: CiRF-ViT and the three reference models.

Each sub-package exposes the same three names, so scripts can treat them
interchangeably:

    get_cfg(dataset)      hyper-parameters for that dataset
    build_model(cfg, ds)  an instantiated model (GP/LN build per cell)
    train(ds, ...)        one training run, written to results/
"""
