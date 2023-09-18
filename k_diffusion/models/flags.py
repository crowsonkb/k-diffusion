import os


def get_use_compile():
    return os.environ.get("K_DIFFUSION_USE_COMPILE", "1") == "1"


def get_use_flash_attention_2():
    return os.environ.get("K_DIFFUSION_USE_FLASH_2", "1") == "1"
