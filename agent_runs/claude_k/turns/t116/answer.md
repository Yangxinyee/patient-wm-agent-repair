Even directly after an action bundle, the state carries the action rates beyond what readings imply only weakly. The posterior z, out and the slot mean all give R² 0.41–0.44 for the current action-A rate, 0.03–0.09 for the previous one, and 0.16–0.26 for action B. The posterior z is already that weak, so the loss happens at or before z_post and the prelude. It is not caused by decay after the bundle.

The next question is whether the action encoder output u_k itself still holds the rates cleanly. If it does (R² near 1), the bottleneck is downstream and I can build a clean channel from u_k into the state. If u_k is already lossy, the encoder needs the edit.

My earlier encoder helpers (`atom_vec`, `enc_a`) can compute u_k offline, and the `act` table seems to list the 2,491 action bundles. I'll inspect both before using them.

```python
print(list(act.columns)); print(act.iloc[1].to_dict())
for fn in ['atom_vec', 'enc_a', 'set_enc', 'enc_act', 'z_act', 'mk_best', 'ctx_of']:
    f = globals()[fn]; sig = inspect.signature(f)
    print(fn, [(k, (type(v.default).__name__ if v.default is not inspect._empty else '-')) for k, v in sig.parameters.items()])
print('fa4 cols', list(fa4.columns), '| de cols', list(de.columns))
```