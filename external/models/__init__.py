def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break


from .Proxy import ProxyRecommender
# from .convmf import ConvMF
# from .convmf import ConvMF

import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        from .convmf.ConvMF import ConvMF
        from .rmg.RMG import RMG
    elif _backend == "pytorch":
        from .ngcf.NGCF import NGCF
        from .lightgcn.LightGCN import LightGCN
        from .egcf.EGCF import EGCF
        from .gat.GAT import GAT
        from .dgcf.DGCF import DGCF

