# Experimental Architectures

This folder contains CNN variants explored during development. They are not used by the main training script but are preserved here for reference and reproducibility.

| File | Description |
|------|-------------|
| `modelcnn1.py` | Deep 4-block CNN with shrinking kernels (32→16→8→4) and growing filters (16→32→64→92). No BatchNorm. |
| `modelcnn3.py` | Shallow single-block CNN with a large 39×39 kernel and sigmoid dense activation. |

## Swapping in an architecture

To run the main script with one of these instead of the default model, update the import in `main_unicnn.py`:

```python
# Replace this line:
from modelcnn_tester import get_model

# With one of:
from architectures.modelcnn1 import get_model
from architectures.modelcnn3 import get_model
```
