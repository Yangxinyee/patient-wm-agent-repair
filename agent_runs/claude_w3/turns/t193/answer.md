The kernel still holds many helpers from earlier sessions. The architecture file is short (5,234 characters, five sections). I will re-read its first half now, input and state update, to rebuild an exact picture of the step computation. That tells me where a dose memory could sit. What I need is a per-bundle update in which some state channel can carry a decaying record of dose amounts.

```python
print(arch[:2700])
```