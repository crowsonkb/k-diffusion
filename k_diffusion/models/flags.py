from contextlib import contextmanager
import os
import threading


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
