from typing import Any, Generic, Protocol, Type, TypeVar, Union, cast, overload
from numpy.typing import NDArray
import numpy as np

from .ragged_buffer import RaggedBufferF32, RaggedBufferI64

ScalarType = TypeVar("ScalarType", bound=np.generic)


class RaggedBuffer(Generic[ScalarType]):
    pass
