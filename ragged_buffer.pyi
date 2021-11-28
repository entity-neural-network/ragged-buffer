from typing import Any, Generic, TypeVar, Union, overload
from numpy.typing import NDArray
import numpy as np

ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)

class RaggedBuffer(Generic[ScalarType]):
    def __init__(self, features: int) -> None: ...
    @classmethod
    def from_array(cls, x: NDArray[ScalarType]) -> RaggedBuffer[ScalarType]: ...
    @classmethod
    def from_flattened(
        cls, flattened: NDArray[ScalarType], lengths: NDArray[np.int64]
    ) -> RaggedBuffer[ScalarType]: ...
    def push(self, x: NDArray[ScalarType]) -> None: ...
    def extend(self, x: RaggedBuffer[ScalarType]) -> None: ...
    def as_array(self) -> NDArray[ScalarType]: ...
    def size0(self) -> int: ...
    def size1(self, i: int) -> int: ...
    def size2(self) -> int: ...
    def __add__(self, other: RaggedBuffer[ScalarType]) -> RaggedBuffer[ScalarType]: ...
    def __getitem__(
        self, i: Union[int, NDArray[np.int64]]
    ) -> RaggedBuffer[ScalarType]: ...
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...

RaggedBufferF32 = RaggedBuffer[np.float32]
RaggedBufferI64 = RaggedBuffer[np.int64]
