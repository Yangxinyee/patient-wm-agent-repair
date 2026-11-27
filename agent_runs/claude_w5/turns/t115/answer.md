The kernel still holds the full reimplementation: `step`, the encoders, and head-related functions (`head_inputs`, `head_probs`, `cmp_head`). fwd_3 is a factual run of the original model. So when the conversation was cut I was probably checking a local value head against fwd_3's predicted means.

Next I will re-read the "Heads" and "How a dose edit is read" sections of ARCHITECTURE.md, plus the source of my head functions, to see what is already verified. If `cmp_head` already reproduces fwd_3's predicted means, I can go straight to designing the edit. If not, I finish the head first.

```python
import inspect
print('\n'.join(lines[54:92])[:3200])
for f in (head_inputs, head_probs, cmp_head):
    s = inspect.getsource(f); print('----', f.__name__, len(s)); print(s[:900])
```