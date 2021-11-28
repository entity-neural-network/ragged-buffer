import numpy as np
import ragged_buffer
from ragged_buffer import RaggedBufferF32

print(ragged_buffer.test("yoyoyo"))

rba = RaggedBufferF32(3)
print(rba)
rba.push(np.array([[2.0, 3.0, 1.0], [1.0, 2.0, 3.0]], dtype=np.float32))
rba.push(
    np.array([[2.0, 3.0, 1.0], [1.0, 2.0, 3.0], [1.4, 2.4, 3.4]], dtype=np.float32)
)
rba.push(
    np.array(
        [[2.0, 3.0, 1.0], [1.0, 2.0, 3.0], [1.4, 2.4, 3.4], [1.4, 2.4, 3.4]],
        dtype=np.float32,
    )
)
rba.push(np.array([[]], dtype=np.float32))
print(rba)

print(rba.as_array())


rba2 = RaggedBufferF32(3)
rba2.push(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32))
rba2.push(
    np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
)

rba.extend(rba2)
print(rba)

assert rba.size0() == 6
assert rba.size1(0) == 2
assert rba.size1(1) == 3
assert rba.size1(2) == 4
assert rba.size1(3) == 0
assert rba.size1(4) == 2
assert rba.size1(5) == 3
assert rba.size2() == 3
