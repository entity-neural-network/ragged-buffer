from typing import Any, Generic, Protocol, Type, TypeVar, Union, cast, overload
from numpy.typing import NDArray
import numpy as np

from .ragged_buffer import RaggedBufferF32, RaggedBufferI64

ScalarType = TypeVar("ScalarType", bound=np.generic)


class RaggedBuffer(Generic[ScalarType]):
    def __init__(self, features: int) -> None:
        raise ValueError(
            "RaggedBuffer is an abstract class, use RaggedBufferF32 or RaggedBufferI64"
        )

    @classmethod
    def from_array(cls, x: NDArray[ScalarType]) -> "RaggedBuffer[ScalarType]":
        if x.dtype == np.float32:
            return RaggedBufferF32.from_array(x)
        elif x.dtype == np.int64:
            return RaggedBufferI64.from_array(x)
        else:
            raise ValueError(
                f"Unsupported dtype {x.dtype}. Only float32 and int64 are currently supported."
            )

    @classmethod
    def from_flattened(
        cls, flattened: NDArray[ScalarType], lengths: NDArray[np.int64]
    ) -> "RaggedBuffer[ScalarType]":
        if flattened.dtype == np.float32:
            return RaggedBufferF32.from_flattened(flattened, lengths)
        elif flattened.dtype == np.int64:
            return RaggedBufferI64.from_flattened(flattened, lengths)
        else:
            raise ValueError(
                f"Unsupported dtype {flattened.dtype}. Only float32 and int64 are currently supported."
            )
