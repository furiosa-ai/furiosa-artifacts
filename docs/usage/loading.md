# Loading models

You can load models provided by Furiosa Artifacts using [furiosa-models](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-models) which based on [furiosa-registry](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-registry).

## Example

### Blocking and Non-Blocking API
Furiosa Model offers both blocking and non-blocking API for fetching models.

#### Blocking example
```python
--8<-- "docs/examples/loading_model.py"
```

#### Non-blocking example
```python
--8<-- "docs/examples/loading_model_nonblocking.py"
```

What's going on here:

`ResNet18(pretrained=True)`

Create model instance. This function ultimately calls the function entrypoint which provided by `artifacts.py` in `furiosa-artifacts`

`pretrained=True` is an arbitrary argument that will transparently pass to model initialization. You can see what arguments are defined in the model class.

---

For non-blocking example

`asyncio.run()`

Function entrypoints in `furiosa.models.nonblocking` are async to support concurrency in loading models. You need to call the entrypoints in async functions or async eventloop.
