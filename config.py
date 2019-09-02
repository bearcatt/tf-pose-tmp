import os
import os.path as osp

import numpy as np

from tfflat.utils import make_dir


class Config:
    ## dataset
    dataset = 'COCO'  # 'COCO', 'PoseTrack', 'MPII'
    testset = 'val'  # train, test, val (there is no validation set for MPII)

    ## directory
    root_dir = osp.dirname(osp.abspath(__file__))
    pretrain_model_dir = osp.join(root_dir, 'dataset', 'imagenet_weights')
    output_dir = osp.join(root_dir, 'output')

    ## model setting
    backbone = 'w32' # 'w32', 'w48'
    init_model = osp.join(pretrain_model_dir, 'hrnet_' + backbone + '.ckpt')
    hrnet_config = osp.join(root_dir, 'hrnet', 'configs', backbone + '.cfg')

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
    bn_train = True
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

    ## to be initialized
    database = None
    num_kps = None
    kps_names = None
    kps_lines = None
    kps_symmetry = None
    img_path = None
    human_det_path = None
    gpu_ids = None
    num_gpus = None
    continue_train = None

    model_str = '{}_{}x{}'.format(backbone, input_shape[0], input_shape[1])
    model_dump_dir = osp.join(output_dir, model_str, 'model_dump', dataset)
    log_dir = osp.join(output_dir, model_str, 'log', dataset)
    result_dir = osp.join(output_dir, model_str, 'result', dataset)

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using /gpu:{}'.format(self.gpu_ids))


def set_dataset_args(cfg):
    datadir = osp.join(cfg.root_dir, 'dataset', cfg.dataset)
    if cfg.dataset == 'COCO':
        from data.coco import database
        database.human_det_path = osp.join(datadir, 'dets', 'human_detection.json')
        database.img_path = osp.join(datadir, 'images')
        database.train_annot_path = osp.join(datadir, 'annotations', 'person_keypoints_train2017.json')
        database.val_annot_path = osp.join(datadir, 'annotations', 'person_keypoints_val2017.json')
        database.test_annot_path = osp.join(datadir, 'annotations', 'image_info_test-dev2017.json')
    elif cfg.dataset == 'MPII':
        from data.mpii import database
        database.human_det_path = osp.join(datadir, 'dets', 'human_detection.json')
        database.img_path = osp.join(datadir)
        database.train_annot_path = osp.join(datadir, 'annotations', 'train.json')
        database.test_annot_path = osp.join(datadir, 'annotations', 'test.json')
    elif cfg.dataset == 'PoseTrack':
        from data.posetrack import database
        database.human_det_path = osp.join(datadir, 'dets', 'human_detection.json')
        database.img_path = osp.join(datadir)
        database.train_annot_path = osp.join(datadir, 'annotations', 'train2018.json')
        database.val_annot_path = osp.join(datadir, 'annotations', 'val2018.json')
        database.test_annot_path = osp.join(datadir, 'annotations', 'test2018.json')
        database.original_annot_path = osp.join(datadir, 'original_annotations')

    cfg.database = database
    cfg.num_kps = database.num_kps
    cfg.kps_names = database.kps_names
    cfg.kps_lines = database.kps_lines
    cfg.kps_symmetry = database.kps_symmetry
    cfg.img_path = database.img_path
    cfg.human_det_path = database.human_det_path


cfg = Config()
set_dataset_args(cfg)

make_dir(cfg.model_dump_dir)
make_dir(cfg.log_dir)
make_dir(cfg.result_dir)
