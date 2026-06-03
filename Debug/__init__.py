from .logger import SimLogger

try:
    from .tb_logger import TBLogger
except ImportError as exc:
    _tb_logger_import_error = exc

    class TBLogger:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TBLogger requires torch.utils.tensorboard or tensorboardX."
            ) from _tb_logger_import_error


__all__ = ["SimLogger", "TBLogger"]
