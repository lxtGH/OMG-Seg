from seg.models.utils.no_obj import NO_OBJ
from seg.models.utils.pan_seg_transform import INSTANCE_OFFSET_HB
from panopticapi.evaluation import PQStat

NO_OBJ_ID = NO_OBJ * INSTANCE_OFFSET_HB


class IoUObj:
    def __init__(self, intersection: int = 0, union: int = 0):
        self.intersection = intersection
        self.union = union

    def __iadd__(self, other):
        self.intersection += other.intersection
        self.union += other.union
        return self

    def __isub__(self, other):
        self.intersection -= other.intersection
        self.union -= other.union
        return self

    def is_legal(self):
        return self.intersection >= 0 and self.union >= 0

    @property
    def iou(self):
        return self.intersection / self.union


def cal_pq(global_intersection_info, classes):
    num_classes = len(classes)
    gt_matched = set()
    pred_matched = set()

    gt_all = set()
    pred_all = set()

    pq_stat = PQStat()
    for gt_id, pred_id in global_intersection_info:
        gt_cat = gt_id // INSTANCE_OFFSET_HB
        pred_cat = pred_id // INSTANCE_OFFSET_HB
        assert pred_cat < num_classes
        if global_intersection_info[gt_id, pred_id].union == 0:
            continue
        if gt_cat == NO_OBJ:
            continue
        gt_all.add(gt_id)
        pred_all.add(pred_id)
        if gt_cat != pred_cat:
            continue
        iou = global_intersection_info[gt_id, pred_id].iou
        if iou > 0.5:
            pq_stat[gt_cat].tp += 1
            pq_stat[gt_cat].iou += iou
            gt_matched.add(gt_id)
            pred_matched.add(pred_id)

    for gt_id in gt_all:
        gt_cat = gt_id // INSTANCE_OFFSET_HB
        if gt_id in gt_matched:
            continue
        pq_stat[gt_cat].fn += 1

    for pred_id in pred_all:
        pred_cat = pred_id // INSTANCE_OFFSET_HB
        if pred_id in pred_matched:
            continue
        if global_intersection_info[NO_OBJ_ID, pred_id].iou > 0.5:
            continue
        pq_stat[pred_cat].fp += 1

    return pq_stat
