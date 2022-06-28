import pathlib

import numpy as np
import yaml

from furiosa.registry import Model

from .base import YoloV5Model, compute_stride
from .box_decode.box_decoder import BoxDecoderC

with open(pathlib.Path(__file__).parent / "datasets/yolov5m/cfg.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    ANCHORS = np.float32(cfg["anchors"])
    CLASS_NAMES = cfg["class_names"]

BOX_DECODER = BoxDecoderC(
    nc=len(CLASS_NAMES),
    anchors=ANCHORS,
    stride=compute_stride(),
    conf_thres=0.25,
)


class YoloV5MediumModel(Model, YoloV5Model):
    def get_class_names(self):
        return CLASS_NAMES

    def get_class_count(self):
        return len(CLASS_NAMES)

    def get_output_feat_count(self):
        return ANCHORS.shape[0]

    def get_anchor_per_layer_count(self):
        return ANCHORS.shape[1]

    def get_box_decoder(self):
        raise BOX_DECODER
