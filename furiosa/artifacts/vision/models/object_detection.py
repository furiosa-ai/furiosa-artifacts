from typing import Any, Sequence, Tuple, List

from furiosa.registry import Model

from .mlcommons.common.datasets import coco, dataset
import numpy as np


class MLCommonsSSDSmallModel(Model):
    """MLCommons MobileNet v1 model"""
    # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/tools/submission/submission-checker.py#L467
    def preprocess(self, image_path: str) -> np.array:
        """Read and preprocess an image located at image_path."""
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/main.py#L49-L51
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/dataset.py#L242-L249
        image = cv2.imread(image_path)
        assert image is not None, image_path
        image = np.array(image, dtype=np.float32)
        if len(image.shape) < 3 or image.shape[2] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_LINEAR)
        image -= 127.5
        image /= 127.5
        image = image.transpose([2, 0, 1])
        return image[np.newaxis, ...]

    def postprocess(self, outputs: 'tensor.TensorArray') -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        outputs = [output.numpy() for output in outputs]
        assert len(outputs) == 12, len(outputs)
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L94-L97
        class_logits = [output.transpose((0, 2, 3, 1)).reshape((1, -1, 91)) for output in outputs[0::2]]
        box_regression = [
            output.transpose((0, 2, 3, 1)).reshape((1, -1, 4)) for output in outputs[1::2]
        ]
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L144-L166
        class_logits = np.concatenate(class_logits, axis=1)  # type: ignore[assignment]
        box_regression = np.concatenate(box_regression, axis=1)  # type: ignore[assignment]
        batch_scores = sigmoid(class_logits)  # type: ignore[arg-type]
        batch_boxes = decode_boxes(box_regression)  # type: ignore[arg-type]

        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L178-L185
        filtered_boxes, filtered_labels, filtered_scores = [], [], []
        for scores, boxes in zip(batch_scores, batch_boxes):
            boxes, labels, scores = filter_results(scores, boxes)
            filtered_boxes.append(boxes)
            filtered_labels.append(labels)
            filtered_scores.append(scores)
        return filtered_boxes, filtered_labels, filtered_scores


class MLCommonsSSDLargeModel(Model):
    """MLCommons ResNet34 model"""

    def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        return dataset.pre_process_coco_resnet34(*args, **kwargs)

    def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        return coco.PostProcessCocoONNXNPlegacy()(*args, **kwargs)
