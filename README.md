Furiosa Artifacts
=================

This repository provides deep learning models provided by FuriosaAI. Available models are described in `artifact.yaml` which is a descriptor file in the form defined by [furiosa-registry](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-registry/)

## Available models

### Image Classification

Model | Python class name | Pretrained? |
:------------ | :-------------|:-------------:|
| ResNet50-v1.5 | `MLCommonsResNet50` | :heavy_check_mark: |
| EfficientNetV2_S | `EfficientNetV2_S` | :heavy_check_mark: |

### Object detection

Model | Python class name | Pretrained? |
:------------ | :-------------|:-------------:|
| SSD-ResNet34 | `MLCommonsSSDResNet34` | :heavy_check_mark: |
| SSD-MobileNets-v1 | `MLCommonsSSDMobilenet` | :heavy_check_mark: |

See `./artifacts.yaml` for more detail.

## Project structure

```bash
.
├── artifacts.yaml                                    (1) Artifact descriptor
├── models                                            (2) Model binary
|   ├── ...
│   └── mlcommons_ssd_resnet34_int8.onnx
├── furiosa
│   └── artifacts
│       └── vision
│           ├── artifacts.py                          (3) Artifact file (for code format)
│           ├── ...
│           └── models                                (4) Model network code
│               ├── __init__.py
│               ├── ...
│               └── mlcommons                         (5) MLCommons specific models
│                   └── ...
│    ...

```

#### Arifact descriptor

A YAML descriptor which has model `artifacts`. `artifact` is a [structure](https://github.com/furiosa-ai/furiosa-sdk/blob/main/python/furiosa-registry/furiosa/registry/artifact.py#L42) defined in `furiosa-registry` to specify **what the model is** and **where it comes from**.

For example, this is a `artifact` in the **artifact descriptor**.

```yaml
  - name: MLCommonsSSDMobileNet
    family: MobileNetV1
    version: v1.1
    location: furiosa/artifacts/artifacts.py
    format: code
    metadata:
      description: MobileNet v1 model for MLCommons v1.1
      publication:
        url: https://arxiv.org/abs/1704.04861.pdf
```

#### Model binary

Model binary in the form of serialized format like [ONNX](https://docs.microsoft.com/en-us/windows/ai/windows-ml/get-onnx-model) or [Tensorflow Lite](https://www.tensorflow.org/lite/guide?hl=ko#1_generate_a_tensorflow_lite_model)

#### Artifact file (for `code` format)

Currently there are two main ways to provide models.

- Providing only serialized model binaries (ONNX, Tflite).
- Providing models via Python code with additional functions (preprocess / postprocess).

This artifact file is to provide models in the form of `code`. `furiosa-registry` can fetch the models from the artifact file.

#### Model network code

Model network code to describe how the model made. This can be [Pytorch module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) or Tensorflow models.

#### MLCommons specific models

This directory is temporarily divided to manage seperately MLCommons model. It requires refactoring later.

## How to upload a new model?

### Providing only serialized binaries.

Upload **model binary** into _models/_.

```bash
furiosa-artifacts$ ls models
mlcommons_resnet50_v1.5_int8.onnx  mlcommons_ssd_mobilenet_v1_int8.onnx  mlcommons_ssd_resnet34_int8.onnx
```

Add metadata into **artifacts descriptor** with `format: onnx` or `format: tflite`. Locate the model via `location` field

```bash
  - name: MLCommonsSSDMobileNet
    family: MobileNetV1
    version: v1.1
    location: models/mlcommons_ssd_mobilenet_v1_int8.onnx
    format: onnx
    metadata:
      description: MobileNet v1 model for MLCommons v1.1
      publication:
        url: https://arxiv.org/abs/1704.04861.pdf
```

### Providing a model described via code.

Add your **model network code** into _furiosa/artifacts/models_.

```python
# furiosa/artifacts/models/vision/image_classification.py

class EfficientNetV2_S(EfficientNetModelGenerator):
    """
    https://arxiv.org/abs/2104.00298
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
    arxiv = "https://arxiv.org/abs/2104.00298"
   ...
```

Add code to **artifact file** which have same name as `name` field in **artifact descriptor**.

```python
# furiosa/artifact/artifacts.py

async def EfficientNetV2_S() -> Model:
    return Model(
        name="efficientnetv2_s",
        model=EfficientNetV2_SModel().export(io.BytesIO()).getvalue(),
    )
```

Upload **model binary** into _models/_ if you have and locate the model like below.

```python
# furiosa/artifact/artifacts.py

async def MLCommonsResNet50() -> MLCommonsResNet50Model:

    return MLCommonsResNet50Model(
        name="mlcommons_resnet50",
        # Note that this code load model from a file
        model=await loader.read("models/mlcommons_resnet50_v1.5_int8.onnx"),
    )
```

Otherwise, you need to instantiate `Model` class with the model in the form of `bytes`.

```python
# furiosa/artifact/artifacts.py

async def EfficientNetV2_S() -> Model:
    return Model(
        name="efficientnetv2_s",
        # Note that this code load model via Python code
        model=EfficientNetV2_SModel().export(io.BytesIO()).getvalue(),
    )
```

Add metadata into **artifacts descriptor** with `format: code` and locate the **artifact file** via `location` field.

```yaml
# artifacts.yaml

  - name: EfficientNetV2_S
    family: EfficientNet
    version: v2.0
    location: furiosa/artifacts/artifacts.py
    format: code
    metadata:
      description: EfficientNetV2 from Google AutoML (https://github.com/google/automl/tree/master/efficientnetv2)
      publication:
        url: https://arxiv.org/abs/2104.00298
```

## How to use models in this repository?

See [client side code](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-registry/#getting-started) in furiosa-reigstry.


## License

```
Copyright (c) 2021 FuriosaAI Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
