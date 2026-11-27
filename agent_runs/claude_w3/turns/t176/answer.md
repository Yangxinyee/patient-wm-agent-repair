The architecture is clear again. My plan is to give each drug a dedicated memory channel inside the model and have the value head read it. Next I'll list the core tensor names and shapes in compact form so I can find exact injection and readout points.

```python
import re
core=[n for n in NAMES if n.startswith('core.')]
print(len(core))
print(' '.join(f"{n[5:]}{list(SHAPES[n])}" for n in core)[:3500])
```