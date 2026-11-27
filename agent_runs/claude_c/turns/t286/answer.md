The key lines show how the record enters the predictions. The value head takes the event's code embedding, a last_numeric term (last value and rate of that code, with its age), and a direct transition table indexed by the last value bin. Actions are predicted from a filtered view built with pending_mlp(p).

Line 50 of the state update ends in "+ (u - (h + E))", and I need to know which u that is. I also need to know whether the dose questions can reach the readings through last_numeric or the transition table. That would happen if z_last_rate of insulin enters the glucose query; if only the event's own code is used, it doesn't.

Next I'll print lines 36-56 of ARCHITECTURE.md, the core update, compactly. If the state update applies the action token directly with no norm, that opens an unsaturated lever. If the only route is the prelude attention, it doesn't.

```python
print('\n'.join(f"{i}: {A[i][:230]}" for i in range(34,57)))
```