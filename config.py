import os
import os.path as osp

import numpy as np

from tfflat.utils import make_dir


class Config:
    ## dataset
    dataset = 'COCO'

    ## directory
    root_dir = osp.dirname(osp.abspath(__file__))
    output_dir = osp.join(root_dir, 'output')

    if dataset == 'COCO':
        import coco
        datadir = osp.join(root_dir, 'dataset/')
        img_path = osp.join(datadir, coco.img_path)
        human_det_path = osp.join(datadir, coco.human_det_path)
        num_kps = coco.num_kps
    else:
        raise NotImplementedError

    ## model setting
    hrnet_size = 32  # 'w32', 'w48'

    ## input, output
    input_shape = (256, 192)  # (256,192), (384,288)
    output_shape = (input_shape[0] // 4, input_shape[1] // 4)
    if output_shape[0] == 64:
        sigma = 2
    elif output_shape[0] == 96:
        sigma = 3
    pixel_means = np.array([[[123.68, 116.78, 103.94]]])

    ## training config
    lr_dec_epoch = [170, 200]
    end_epoch = 210
    lr = 1e-3
    lr_dec_factor = 10
    optimizer = 'adam'
    weight_decay = 1e-5
    batch_size = 32
    scale_factor = 0.35
    rotation_factor = 45

    ## testing config
    useGTbbox = False
    flip_test = True
    oks_nms_thr = 0.9
    score_thr = 0.2
    test_batch_size = 32

    ## others
    multi_thread_enable = True
    num_thread = 10
    log_display = 1

    model_str = 'w{}_{}x{}'.format(hrnet_size, input_shape[0], input_shape[1])
    model_dump_dir = osp.join(output_dir, model_str, 'model_dump', dataset)
    log_dir = osp.join(output_dir, model_str, 'log', dataset)
    result_dir = osp.join(output_dir, model_str, 'result', dataset)

    def set_args(self, gpu_ids, weights):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.weights = weights
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using /gpu:{}'.format(self.gpu_ids))


cfg = Config()

make_dir(cfg.model_dump_dir)
make_dir(cfg.log_dir)
make_dir(cfg.result_dir)


import tensorflow as tf

tf.app.flags.DEFINE_string('dataset', 'COCO', 'The dataset to use.')

tf.app.flags.DEFINE_string('root_dir', '.', 'The root directory.')

tf.app.flags.DEFINE_string('output_dir', 'output', 'The output directory.')

tf.app.flags.DEFINE_string('datadir', 'dataset', 'The dataset directory.')

tf.app.flags.DEFINE_string('img_path', 'dataset/images', 'The image path.')

tf.app.flags.DEFINE_string('human_det_path', 'dataset/human_detections.json', 'The detected human boxes.')

tf.app.flags.DEFINE_integer('num_kps', 17, 'Number of key points.')


