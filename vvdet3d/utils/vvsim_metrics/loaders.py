# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.

import json
from typing import Dict, Tuple

import numpy as np
import tqdm
from pyquaternion import Quaternion

from nuscenes import NuScenes
from .common.data_classes import EvalBoxes
from .detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.splits import create_splits_scenes

def yaw2quaternion(yaw: float) -> list:
    qw = np.cos(yaw / 2)
    qx = 0
    qy = 0
    qz = np.sin(yaw / 2)
    return [qw, qx, qy, qz]    


def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) \
        -> Tuple[EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data['meta']
    if verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(all_results.sample_idxes)))
    return all_results, meta


def load_gt(data_info: dict, box_cls, verbose: bool = False) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param data_info: A dict containing dataset information.
    :param ego_vehicle: The name of the ego vehicle in the dataset, e.g.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    # Init.
    # Read out all sample_tokens in DB.
    sample_idxes_all = [i for i in range(data_info['metainfo']['scene_length'])]
    assert len(sample_idxes_all) > 0, "Error: Database has no samples!"

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    sample_idx = -1
    for scene_info in data_info['scene_list']:
        for sample_info in scene_info['data_list']:
            sample_idx += 1
            data_dict = sample_info['data_dict']
            sample_boxes = []
            for instance_global in data_dict['instances_global']:
                if box_cls == DetectionBox:
                    detection_name = instance_global['label_name']
                    bbox3d = instance_global['bbox_3d']
                    sample_boxes.append(
                        box_cls(
                            sample_idx=sample_idx,
                            translation=[bbox3d[0], bbox3d[1], bbox3d[2]],
                            size=[bbox3d[4], bbox3d[3], bbox3d[5]],
                            rotation=yaw2quaternion(bbox3d[6]),
                            velocity=[bbox3d[7], bbox3d[8]],
                            detection_name=detection_name,
                            detection_score=-1.0,  # GT samples do not have a score.
                        )
                    )
    
    
                else:
                    raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

            all_annotations.add_boxes(sample_idx, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations



def filter_eval_boxes(eval_boxes: EvalBoxes,
                      max_dist: Dict[str, float],
                      verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field = _get_box_class_field(eval_boxes)

    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
    for ind, sample_idx in enumerate(eval_boxes.sample_idxes):

        # Filter on distance first.
        total += len(eval_boxes[sample_idx])
        eval_boxes.boxes[sample_idx] = [box for box in eval_boxes[sample_idx] if
                                          box.ego_dist < max_dist[box.__getattribute__(class_field)] and box.ego_dist > 0.01]
        dist_filter += len(eval_boxes[sample_idx])

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)
        print("=> After bike rack filtering: %d" % bike_rack_filter)

    return eval_boxes


def _get_box_class_field(eval_boxes: EvalBoxes) -> str:
    """
    Retrieve the name of the class field in the boxes.
    This parses through all boxes until it finds a valid box.
    If there are no valid boxes, this function throws an exception.
    :param eval_boxes: The EvalBoxes used for evaluation.
    :return: The name of the class field in the boxes, e.g. detection_name or tracking_name.
    """
    assert len(eval_boxes.boxes) > 0
    box = None
    for val in eval_boxes.boxes.values():
        if len(val) > 0:
            box = val[0]
            break
    if isinstance(box, DetectionBox):
        class_field = 'detection_name'
    elif isinstance(box, TrackingBox):
        class_field = 'tracking_name'
    else:
        raise Exception('Error: Invalid box type: %s' % box)

    return class_field
