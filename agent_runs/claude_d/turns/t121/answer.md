The kernel still holds many useful helpers: `head`, `headv`, `mlp`, `true_shift_ins`, `true_shift_ne`, `stay_panel` and others. The architecture document is only 5.2k characters, so I'll print it in full. I need the exact head inputs and the recurrence details to choose where a centred dose correction can go: whether `out` feeds back, how the z prior is used, and how NLL is computed. Nothing is being tested yet; this sets up the edit design.

```python
print(A)
```