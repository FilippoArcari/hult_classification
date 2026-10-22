import torch
from lightning.pytorch.loggers import TensorBoardLogger


class InspectionTensorBoardLogger(TensorBoardLogger):
    def log_graph(self, model, input_array=None):
        # The Transformer fused eval path can differ between JIT's tracing and
        # verification passes. Keep the same attention path for this export only.
        fastpath = torch.backends.mha.get_fastpath_enabled()
        try:
            torch.backends.mha.set_fastpath_enabled(False)
            return super().log_graph(model, input_array)
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)
