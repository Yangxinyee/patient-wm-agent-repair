Run 212 shows the architecture file is short and has a section on the core step. Next I will print that section only. I want the exact update for h, `c_out` and the coda, to see whether a dose-carrying direction can pass from one step to the next without being normalised away. A residual coda with no final norm would confirm that a dedicated channel is possible. A final normalisation on h would argue against it.

```python
i=A_.find('## State'); j=A_.find('## Heads')
print(A_[i:j][:2200])
```