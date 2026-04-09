"""
GPU 配置模块：自动检测并配置 TensorFlow GPU 环境。

在所有训练/预测脚本的最开头 import 此模块:
    from deeponet_pbe.gpu_config import setup_gpu
    setup_gpu()
"""

import os


def setup_gpu(memory_growth=True, memory_limit_mb=None, visible_gpus="0"):
    """配置 TensorFlow GPU 环境。

    Parameters
    ----------
    memory_growth : bool
        启用显存按需增长（推荐），防止一次占满所有显存。
    memory_limit_mb : int | None
        限制 GPU 最大显存使用 (MB)。None 则不限制。
    visible_gpus : str
        可见 GPU 编号，如 "0" 或 "0,1"。仅当环境中尚未设置
        ``CUDA_VISIBLE_DEVICES`` 时写入该变量，以便 Conda 环境
        （如 ``deeponet_gpu``）在 activate 脚本里预先指定显卡。
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_gpus)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print("[GPU] No GPU found, running on CPU.")
        return

    print(f"[GPU] Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu.name}")

    for gpu in gpus:
        if memory_limit_mb is not None:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(
                    memory_limit=memory_limit_mb
                )],
            )
            print(f"[GPU] Memory limit set to {memory_limit_mb} MB")
        elif memory_growth:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("[GPU] Memory growth enabled (allocate on demand)")

    print("[GPU] Configuration complete.")
