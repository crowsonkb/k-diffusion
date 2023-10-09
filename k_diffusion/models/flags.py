from contextlib import contextmanager
from functools import update_wrapper
import os
import threading

import torch


def get_use_compile():
    return os.environ.get("K_DIFFUSION_USE_COMPILE", "1") == "1"


def get_use_flash_attention_2():
    return os.environ.get("K_DIFFUSION_USE_FLASH_2", "1") == "1"


state = threading.local()
state.checkpointing = False


@contextmanager
def checkpointing(enable=True):
    try:
        old_checkpointing, state.checkpointing = state.checkpointing, enable
        yield
    finally:
        state.checkpointing = old_checkpointing


def get_checkpointing():
    return getattr(state, "checkpointing", False)


class compile_wrap:
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._compiled_function = None
        update_wrapper(self, function)

    @property
    def compiled_function(self):
        if self._compiled_function is not None:
            return self._compiled_function
        if get_use_compile():
            try:
                self._compiled_function = torch.compile(self.function, *self.args, **self.kwargs)
            except RuntimeError:
                self._compiled_function = self.function
        else:
            self._compiled_function = self.function
        return self._compiled_function

    def __call__(self, *args, **kwargs):
        return self.compiled_function(*args, **kwargs)
