The bytecode shows what the lost helpers do:
- `kill_dims` only zeroes the end-of-step writers of a dim: the `c_out` rows and the coda output rows and biases. Readers stay connected, so it is not a full repurposing kill.
- `rollout_gen`/`fn` are my offline dose-edit rollout. It keeps the record to hour 12, starts drawing readings at minute 780 (hour 13) and applies rate/dose scaling from hour 12 on.

ARCHITECTURE.md says `filter_gate`/`pending_mlp` only feed the action heads through a filtered view, so they are off the value-prediction path.

The file is only 5,234 characters, so I'll print all of it. I want the exact formulas for the recurrence (`a_in`, `delta_in`, `b_in`, `flow`), the prelude and coda layers, and the value head's inputs. That will let me write a complete repurposing kill: cut every writer and every reader of a chosen dim. If the file matches my notes, I'll write that kill function next and measure its joint cost on many stays.

```python
print(open('/model/ARCHITECTURE.md').read())
```