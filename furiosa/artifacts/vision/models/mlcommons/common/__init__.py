from typing import Any, Optional

from furiosa.registry import Model


class SessionBaseModel(Model):
    sess: Optional[Any] = None
