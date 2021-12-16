from typing import Any, Generic, List, Protocol, Type, TypeVar, Union, cast, overload
from numpy.typing import NDArray
import numpy as np

from .ragged_buffer import RaggedBufferF32, RaggedBufferI64, RaggedBufferBool

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
        elif x.dtype == np.bool_:
            return RaggedBufferBool.from_array(x)
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
        elif flattened.dtype == np.bool_:
            return RaggedBufferBool.from_flattened(flattened, lengths)
        else:
            raise ValueError(
                f"Unsupported dtype {flattened.dtype}. Only float32 and int64 are currently supported."
            )


def cat(
    buffers: List[RaggedBuffer[ScalarType]], dim: int = 0
) -> RaggedBuffer[ScalarType]:
    if len(buffers) == 0:
        raise ValueError("Can't concatenate an empty list of buffers")
    else:
        if isinstance(buffers[0], RaggedBufferF32):
            return RaggedBufferF32.cat(buffers, dim)
        elif isinstance(buffers[0], RaggedBufferI64):
            return RaggedBufferI64.cat(buffers, dim)
        elif isinstance(buffers[0], RaggedBufferBool):
            return RaggedBufferBool.cat(buffers, dim)
        else:
            raise TypeError(f"Type {type(buffers[0])} is not a RaggedBuffer")
