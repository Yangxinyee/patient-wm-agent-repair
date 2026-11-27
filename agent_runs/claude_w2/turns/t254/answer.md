Now I can read the pieces. `gen_wc` collects the value head's with_code vectors in teacher-forced runs where only the doses are changed, together with each row's true lagged doses. Run 244 was therefore fitting decoders that read recent insulin doses off head features ("base", "wc", "wcm"). It then checked how far the decoded dose moves when the dose is changed counterfactually (the "cf" ratios, likely one per lag).

The weak ratios (lag 1 only 0.03–0.15) suggest these decoders mostly track glucose level, not the dose actually given. That would make a head-only readout fail in dose_edit rollouts. `linor.py` is the oracle hook that tilts the predicted bin distribution by an exact amount computed from the lagged doses. `corechan.npz` (wI48, vI, vN, nsc) looks like an attempt to inject a dose channel into the core state.

The kernel functions have no source, so the fastest way to recover runs 208–244 is the interactive input history, if this kernel keeps one. If it does, I'll read the code of the last runs. Otherwise I'll redesign from the documented facts.

```python
has = {k:(k in globals()) for k in ['In','_ih','Out','get_ipython']}
print(has)
H = globals().get('In') or globals().get('_ih')
if H:
    print("n entries", len(H))
    for i in range(max(1,len(H)-40), len(H)-3):
        print(f"--[{i}] {len(H[i])} chars: ", H[i][:160].replace('\n',' | '))
    for i in range(len(H)-3, len(H)):
        print(f"=====[{i}]\n", H[i][:2500])
```