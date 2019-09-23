import json
import os.path as osp
import pickle

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

num_kps = 17
kps_symmetry = [(1, 2), (3, 4), (5, 6), (7, 8),
                (9, 10), (11, 12), (13, 14), (15, 16)]
train_annot_path = 'annotations/person_keypoints_train2017.json'
val_annot_path = 'annotations/person_keypoints_val2017.json'
test_annot_path = 'annotations/image_info_test-dev2017.json'
human_det_path = 'dets/human_detection.json'
img_path = 'images'

testset = 'val'


def load_train_data(datadir, score=False):
    coco = COCO(osp.join(datadir, train_annot_path))
    train_data = []
    for aid in coco.anns.keys():
        ann = coco.anns[aid]
        imgname = 'train2017/' + coco.imgs[ann['image_id']]['file_name']
        joints = ann['keypoints']

        if (ann['image_id'] not in coco.imgs) or ann['iscrowd'] or \
                np.sum(joints[2::3]) == 0 or ann['num_keypoints'] == 0:
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
            data = dict(
                image_id=ann['image_id'], imgpath=imgname,
                bbox=bbox, joints=joints, score=1)
        else:
            data = dict(
                image_id=ann['image_id'], imgpath=imgname,
                bbox=bbox, joints=joints)

        train_data.append(data)

    return train_data


def load_annot(annot_path):
    return COCO(osp.join(datadir, annot_path))


def imgid_to_imgname(annot, imgid, db_set):
    imgs = annot.loadImgs(imgid)
    imgname = [db_set + '2017/' + i['file_name'] for i in imgs]
    return imgname


def evaluation(result, gt, result_dir, db_set):
    result_path = osp.join(result_dir, 'result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f)

    result = gt.loadRes(result_path)
    cocoEval = COCOeval(gt, result, iouType='keypoints')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    result_path = osp.join(result_dir, 'result.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(cocoEval, f, 2)
        print("Saved result file to " + result_path)
