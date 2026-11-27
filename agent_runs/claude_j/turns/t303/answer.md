The core section shows h is updated additively, `h ← a_bar·h + softplus(delta_in)·b_in(E) + (u − (h+E))`, and then goes through a "coda" built from `c_out`. Whether a dose sum can survive from step to step depends on what the coda does: a plain linear residual would preserve an added dimension, a normalisation would squash it. I'll print the rest of the core section (lines 30–53) in short form to check.

```python
s='\n'.join(f"{i}:{L[i].strip()[:210]}" for i in range(30,54) if L[i].strip())
print(s[:4000])
```