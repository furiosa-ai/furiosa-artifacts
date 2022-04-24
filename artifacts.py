import io
from typing import Any

from furiosa.registry import Format, Metadata, Model, Publication
from furiosa.registry.client.transport import FileTransport

from furiosa.artifacts.vision.models.image_classification import (
    EfficientNetV2_S as EfficientNetV2_SModel,
)
from furiosa.artifacts.vision.models.image_classification import MLCommonsResNet50Model
from furiosa.artifacts.vision.models.object_detection import (
    MLCommonsSSDLargeModel,
    MLCommonsSSDSmallModel,
)

loader = FileTransport()


# Image classification


async def MLCommonsResNet50(*args: Any, **kwargs: Any) -> MLCommonsResNet50Model:
    return MLCommonsResNet50Model(
        name="MLCommonsResNet50",
        model=await loader.read("models/mlcommons_resnet50_v1.5_int8.onnx"),
        format=Format.ONNX,
        family="ResNet",
        version="v1.1",
        metadata=Metadata(
            description="ResNet50 v1.5 model for MLCommons v1.1",
            publication=Publication(url="https://arxiv.org/abs/1512.03385.pdf"),
        ),
        **kwargs,
    )


async def EfficientNetV2_S(*args: Any, **kwargs: Any) -> Model:
    return Model(
        name="EfficientNetV2_S",
        model=EfficientNetV2_SModel().export(io.BytesIO()).getvalue(),
        format=Format.ONNX,
        family="EfficientNet",
        version="v2.0",
        metadata=Metadata(
            description="EfficientNetV2 from Google AutoML",
            publication=Publication(url="https://arxiv.org/abs/2104.00298"),
        ),
        **kwargs,
    )


# Object detection


async def MLCommonsSSDMobileNet(*args: Any, **kwargs: Any) -> MLCommonsSSDSmallModel:
    return MLCommonsSSDSmallModel(
        name="MLCommonsSSDMobileNet",
        model=await loader.read("models/mlcommons_ssd_mobilenet_v1_int8.onnx"),
        format=Format.ONNX,
        family="MobileNetV1",
        version="v1.1",
        metadata=Metadata(
            description="MobileNet v1 model for MLCommons v1.1",
            publication=Publication(url="https://arxiv.org/abs/1704.04861.pdf"),
        ),
        **kwargs,
    )


async def MLCommonsSSDResNet34(*args: Any, **kwargs: Any) -> MLCommonsSSDLargeModel:
    return MLCommonsSSDLargeModel(
        name="MLCommonsSSDResNet34",
        model=await loader.read("models/mlcommons_ssd_resnet34_int8.onnx"),
        format=Format.ONNX,
        family="ResNet",
        version="v1.1",
        metadata=Metadata(
            description="ResNet34 model for MLCommons v1.1",
            publication=Publication(
                url="https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection"  # noqa: E501
            ),
        ),
        **kwargs,
    )
