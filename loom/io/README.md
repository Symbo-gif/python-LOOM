# loom I/O Module

**Save/load tensors in loom format**

## Status: ✅ COMPLETE (Phase 4)

## Features

- `save(filename, tensor)`: Save a tensor to `.tfdata` format.
- `load(filename)`: Load a tensor from `.tfdata` format.

## File Format

The `.tfdata` format is a simple JSON-based serialization:
- `shape`: List of dimension sizes.
- `dtype`: String representation of the data type.
- `data`: Flattened list of values.

## Usage Example

```python
import loom as lm
from loom.io import save, load

# Save
a = lm.array([[1, 2], [3, 4]])
save('my_tensor.tfdata', a)

# Load
b = load('my_tensor.tfdata')
print(b.tolist())  # [[1.0, 2.0], [3.0, 4.0]]
```

## Notes

- Complex numbers are serialized as `[real, imag]` pairs.
- Symbolic tensors cannot be saved (data must be computed first).
