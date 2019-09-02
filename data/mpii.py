#!/usr/bin/python3
# coding=utf-8

import os.path as osp

import numpy as np
from pycocotools.coco import COCO
from scipy.io import savemat


class Dataset(object):
    num_kps = 16
    kps_names = ['r_ankle', 'r_knee', 'r_hip', 'l_hip', 'l_knee',
                 'pelvis', 'throax', 'upper_neck', 'head_top', 'r_wrist',
                 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']
    kps_symmetry = [(0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13)]
    kps_lines = [(0, 1), (1, 2), (2, 6), (7, 12), (12, 11), (11, 10), (5, 4),
                 (4, 3), (3, 6), (7, 13), (13, 14), (14, 15), (6, 7), (7, 8), (8, 9)]
    train_annot_path = None # initialized in config.py
    val_annot_path = None
    test_annot_path = None

    def load_train_data(self, score=False):
        coco = COCO(self.train_annot_path)
        train_data = []
        for aid in coco.anns.keys():
            ann = coco.anns[aid]
            imgname = coco.imgs[ann['image_id']]['file_name']
            joints = ann['keypoints']

            if (ann['image_id'] not in coco.imgs) or ann['iscrowd'] or (ann['num_keypoints'] == 0):
                continue

            # sanitize bboxes
            x, y, w, h = ann['bbox']
            img = coco.loadImgs(ann['image_id'])[0]
            width, height = img['width'], img['height']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if ann['area'] > 0 and x2 >= x1 and y2 >= y1:
                bbox = [x1, y1, x2 - x1, y2 - y1]
            else:
                continue

            if score:
                data = dict(image_id=ann['image_id'], imgpath=imgname, bbox=bbox, joints=joints, score=1)
            else:
                data = dict(image_id=ann['image_id'], imgpath=imgname, bbox=bbox, joints=joints)
            train_data.append(data)

        return train_data

    def load_annot(self, db_set):
        if db_set == 'train':
            coco = COCO(self.train_annot_path)
        elif db_set == 'test':
            coco = COCO(self.test_annot_path)
        else:
            print('Unknown db_set')
            assert 0

        return coco

    def load_imgid(self, annot):
        return annot.imgs

    def imgid_to_imgname(self, annot, imgid, db_set):
        imgs = annot.loadImgs(imgid)
        imgname = [i['file_name'] for i in imgs]
        return imgname

    def evaluation(self, result, annot, result_dir, db_set):
        result_path = osp.join(result_dir, 'result.mat')
        savemat(result_path, mdict=result)


database = Dataset()
