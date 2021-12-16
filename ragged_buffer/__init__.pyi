import numpy as np
from typing import Any, Generic, TypeVar, Union, overload, List
from numpy.typing import NDArray
import numpy as np

ScalarType = TypeVar("ScalarType", bound=np.generic)

class RaggedBuffer(Generic[ScalarType]):
    def __init__(self, features: int) -> None: ...
    @classmethod
    def from_array(cls, x: NDArray[ScalarType]) -> RaggedBuffer[ScalarType]: ...
    @classmethod
    def from_flattened(
        cls, flattened: NDArray[ScalarType], lengths: NDArray[np.int64]
    ) -> RaggedBuffer[ScalarType]: ...
    def push(self, x: NDArray[ScalarType]) -> None: ...
    def push_empty(self) -> None: ...
    def extend(self, x: RaggedBuffer[ScalarType]) -> None: ...
    def as_array(self) -> NDArray[ScalarType]: ...
    def size0(self) -> int: ...
    @overload
    def size1(self) -> NDArray[np.int64]: ...
    @overload
    def size1(self, i: int) -> int: ...
    def size2(self) -> int: ...
    @overload
    def __add__(self, other: RaggedBuffer[ScalarType]) -> RaggedBuffer[ScalarType]: ...
    @overload
    def __add__(self, other: int) -> RaggedBuffer[ScalarType]: ...
    @overload
    def __add__(self, other: float) -> RaggedBuffer[np.float32]: ...
    @overload
    def __mul__(self, other: RaggedBuffer[ScalarType]) -> RaggedBuffer[ScalarType]: ...
    @overload
    def __mul__(self, other: int) -> RaggedBuffer[ScalarType]: ...
    @overload
    def __mul__(self, other: float) -> RaggedBuffer[np.float32]: ...
    def __getitem__(
        self, i: Union[int, NDArray[np.int64]]
    ) -> RaggedBuffer[ScalarType]: ...
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...
    def clear(self) -> None: ...
    def indices(self, dim: int) -> RaggedBufferI64: ...
    def flat_indices(self) -> RaggedBufferI64: ...

RaggedBufferF32 = RaggedBuffer[np.float32]
RaggedBufferI64 = RaggedBuffer[np.int64]
RaggedBufferBool = RaggedBuffer[np.bool_]

def cat(
    buffers: List[RaggedBuffer[ScalarType]], dim: int = 0
) -> RaggedBuffer[ScalarType]: ...
