import io
from typing import Any

from furiosa.registry import Model
from furiosa.registry.client.transport import FileTransport

from .vision.models.image_classification import (
    EfficientNetV2_S as EfficientNetV2_SModel,
)
from .vision.models.mlcommons.common.datasets import coco, dataset

loader = FileTransport()


class MLCommonsResNet50Model(Model):
    def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        return dataset.pre_process_vgg(*args, **kwargs)

    def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        return dataset.PostProcessArgMax(offset=0)


class MLCommonsSSDSmallModel(Model):
    def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        return dataset.pre_process_coco_pt_mobilenet(*args, **kwargs)

    def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        return coco.PostProcessCocoSSDMobileNetORTlegacy(False, 0.3)(*args, **kwargs)


class MLCommonsSSDLargeModel(Model):
    def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        return dataset.pre_process_coco_resnet34(*args, **kwargs)

    def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        return coco.PostProcessCocoONNXNPlegacy()(*args, **kwargs)


# Image classification


async def MLCommonsResNet50(*args: Any, **kwargs: Any) -> MLCommonsResNet50Model:

    return MLCommonsResNet50Model(
        name="MLCommonsResNet50",
        model=await loader.read("models/mlcommons_resnet50_v1.5_int8.onnx"),
        **kwargs,
    )


async def EfficientNetV2_S(*args: Any, **kwargs: Any) -> Model:
    return Model(
        name="EfficientNetV2_S",
        model=EfficientNetV2_SModel().export(io.BytesIO()).getvalue(),
        **kwargs,
    )


# Object detection


async def MLCommonsSSDMobileNet(*args: Any, **kwargs: Any) -> MLCommonsSSDSmallModel:

    return MLCommonsSSDSmallModel(
        name="MLCommonsSSDMobileNet",
        model=await loader.read("models/mlcommons_ssd_mobilenet_v1_int8.onnx"),
        **kwargs,
    )


async def MLCommonsSSDResNet34(*args: Any, **kwargs: Any) -> MLCommonsSSDLargeModel:

    return MLCommonsSSDLargeModel(
        name="MLCommonsSSDResNet34",
        model=await loader.read("models/mlcommons_ssd_resnet34_int8.onnx"),
        **kwargs,
    )
