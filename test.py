import numpy as np
import ragged_buffer
from ragged_buffer import RaggedBuffer

print(ragged_buffer.test("yoyoyo"))

rba = RaggedBuffer(3)
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
