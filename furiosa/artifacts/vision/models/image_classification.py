from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import timm
from furiosa.registry import Model
from furiosa.runtime import tensor

from .common.base import ImageNetRwightman
from .mlcommons.common.utils import mixin
from .mlcommons.common.datasets import imagenet1k


class SessionBaseModel(Model):
    sess: Optional[Any] = None


def call_model_generator(parent_class) -> Callable:
    class ModelGenerator(parent_class):
        configer = parent_class.configer

        def __init__(self, pretrained: Optional[bool] = False):
            self.check_validity(pretrained)
            if not pretrained:
                self.training_data = "none"
            super(ModelGenerator, self).__init__(
                model_func=self.model_func,
                input_shape=self.input_shape,
                pretrained=pretrained,
            )

        def check_validity(self, pretrained):
            if pretrained:
                if not self.has_pretrained:
                    raise Exception(
                        "%s has no trained weights." % self.__class__.__name__
                    )

    return ModelGenerator


TimmModelGenerator = call_model_generator(ImageNetRwightman)


class EfficientNetModelGenerator(TimmModelGenerator):
    module_name = "efficientnet"

    @staticmethod
    def configer(model_name, config_key=None):
        if not config_key:
            config_key = model_name
        return TimmModelGenerator.configer(
            module=getattr(timm.models, EfficientNetModelGenerator.module_name),
            model_name=model_name,
            config_key=config_key,
        )


class EfficientNetV2_S(EfficientNetModelGenerator):
    """EfficientNet V2 model

    https://github.com/google/automl/tree/master/efficientnetv2
    """

    model_family = "EffcientNetV2"
    model_name = "efficientnetv2_s"
    config_key = "tf_efficientnetv2_s"
    (
        model_config,
        model_func,
        input_shape,
        has_pretrained,
    ) = EfficientNetModelGenerator.configer(config_key)


class EfficientNetV2_M(EfficientNetV2_S):
    """EfficientNet V2 model

    https://github.com/google/automl/tree/master/efficientnetv2
    """

    model_name = "efficientnetv2_m"
    config_key = "tf_efficientnetv2_m"
    (
        model_config,
        model_func,
        input_shape,
        has_pretrained,
    ) = EfficientNetModelGenerator.configer(config_key)


class MLCommonsResNet50Model(mixin.SessionMixin, SessionBaseModel):
    """MLCommons ResNet50 model"""

    idx2str: List[str] = imagenet1k.ImageNet1k_Idx2Str

    def center_crop(
        self, image: np.ndarray, cropped_height: int, cropped_width: int
    ) -> np.ndarray:
        """Centrally crop `image` into cropped_width x cropped_height."""
        height, width, _ = image.shape
        top = int((height - cropped_height) / 2)
        bottom = int((height + cropped_height) / 2)
        left = int((width - cropped_width) / 2)
        right = int((width + cropped_width) / 2)
        image = image[top:bottom, left:right]
        return image

    def resize_with_aspect_ratio(
        self,
        image: np.ndarray,
        scaled_height: int,
        scaled_width: int,
        percent: float,
        interpolation: int,
    ) -> np.ndarray:
        """Resize `image` so that it will be of scaled_width x scaled_height if it is scaled by `percent`."""
        height, width, _ = image.shape
        new_height = int(100.0 * scaled_height / percent)
        new_width = int(100.0 * scaled_width / percent)
        if height > width:
            new_height = int(new_height * height / width)
        else:
            new_width = int(new_width * width / height)
        image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        return image

    def preprocess(self, image_path: str) -> tensor.TensorArray:
        """Read and preprocess an image located at image_path."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.resize_with_aspect_ratio(
            image, 224, 224, percent=87.5, interpolation=cv2.INTER_AREA
        )
        image = self.center_crop(image, 224, 224)
        image = np.asarray(image, dtype=np.float32)
        image -= np.array([123.68, 116.78, 103.94], dtype=np.float32)
        image = image.transpose([2, 0, 1])
        return image[np.newaxis, ...]

    def postprocess(self, output: np.ndarray) -> str:
        return self.idx2str[int(output[0].numpy()) - 1]

    def inference(self, image: np.array) -> np.array:
        return self.sess.run(image)

    def __call__(self, image_path: str) -> Dict[int, str]:
        return self.postprocess(self.inference(self.preprocess(image_path)))
