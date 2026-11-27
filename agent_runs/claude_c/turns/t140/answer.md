Restarted with my notes in hand. The op list still holds `scale core.prelude.0.attn.out.weight` from passes 11/12. The notebook rejected that edit because it costs +0.017 NLL, so I need to remove it. Before choosing the next lever I'll re-read the architecture and the tensor list. Every route I tried failed: input scaling saturated, z was impotent, and the state/head remaps were not selective. I need to find where the action token's contribution can be scaled without touching the reading pathway. I'll clear the stale op in this run too.

```python
clear_ops()
print(ops())
txt = open('/model/ARCHITECTURE.md').read()
print(len(txt))
print(txt[:12000])
```