def save_import_tensorflow(gpu: str) -> 'tensorflow':
    import sys
    import os

    if 'tensorflow' in sys.modules or 'torch' in sys.modules:
        print('Tensorflow already imported.',
              f'Number of available GPU(s): {len(os.environ["CUDA_VISIBLE_DEVICES"])}',
              f'ID(s) for visible GPU(s): {os.environ["CUDA_VISIBLE_DEVICES"]}', sep=os.linesep)
        return sys.modules['tensorflow']

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print('Imported Tensorflow.',
          f'ID(s) for visible GPU(s): {os.environ["CUDA_VISIBLE_DEVICES"]}', sep=os.linesep)
    import tensorflow as tf
    return tf


