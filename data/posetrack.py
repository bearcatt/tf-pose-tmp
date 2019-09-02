#!/usr/bin/python3
# coding=utf-8

import glob
import json
import os.path as osp

import numpy as np
from pycocotools.coco import COCO


class Dataset(object):
    num_kps = 17
    kps_names = ['nose', 'head_bottom', 'head_top', 'l_ear', 'r_ear', 'l_shoulder',
                 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    kps_symmetry = [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    kps_lines = [(0, 1), (0, 2), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14),
                 (14, 16), (11, 13), (13, 15), (5, 6), (11, 12)]
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

    def load_val_data_with_annot(self):
        coco = COCO(self.val_annot_path)
        val_data = []
        for aid in coco.anns.keys():
            ann = coco.anns[aid]
            if ann['image_id'] not in coco.imgs:
                continue
            imgname = coco.imgs[ann['image_id']]['file_name']
            bbox = ann['bbox']
            joints = ann['keypoints']
            data = dict(image_id=ann['image_id'], imgpath=imgname, bbox=bbox, joints=joints, score=1)
            val_data.append(data)

        return val_data

    def load_annot(self, db_set):
        if db_set == 'train':
            coco = COCO(self.train_annot_path)
        elif db_set == 'val':
            coco = COCO(self.val_annot_path)
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
        # convert coco format to posetrack format
        # evaluation is available by poseval (https://github.com/leonid-pishchulin/poseval)

        print('Converting COCO format to PoseTrack format...')
        filenames = glob.glob(osp.join(self.original_annot_path, db_set, '*.json'))
        for i in range(len(filenames)):

            with open(filenames[i]) as f:
                annot = json.load(f)
            img_id_list = []
            for ann in annot['images']:
                img_id_list.append(ann['id'])

            dump_result = {}
            dump_result['images'] = annot['images']
            dump_result['categories'] = annot['categories']
            annot_from_result = []
            for res in result:
                if res['image_id'] in img_id_list:
                    annot_from_result.append(res)
            dump_result['annotations'] = annot_from_result

            result_path = osp.join(result_dir, filenames[i].split('/')[-1])
            with open(result_path, 'w') as f:
                json.dump(dump_result, f)


database = Dataset()
