The best h direction gives a generalised SNR of only 0.17. The dose imprint is buried in natural step variation, so amplifying in state space would scale up everything else too. Input-side scaling saturates, and the z route is impotent.

Drug information can only reach the state through the prelude's attention to the action token u. If readings enter mainly through memory cross-attention instead, then scaling the prelude attention output would act mostly on drugs. I'll test potency with `scale(core.prelude.0.attn.out.weight, 3)`:
- if the four readouts grow roughly 3×, this is a viable lever, and I'll check factual cost next;
- if they barely change, the prelude is not the bottleneck.

```python
clear_ops()
scale('core.prelude.0.attn.out.weight', 3.0)
print(ops())
with open('/work/notes.md','a') as f:
    f.write("- Run109: best h-dir generalized SNR (dose-noise imprint at ins bundles vs reading-step cov) = 0.17 -> h-space amplification non-selective.\n- Pass 11 plan: scale prelude.attn.out x3 (only potent drug path is prelude attention to u token).\n")
```