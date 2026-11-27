Even with the full norm1 input, an attention-value decoder cannot both read the start rate accurately and stay near zero on the other tokens. The best on-target error was 0.31. The rate direction overlaps with the content of the e, slot and z tokens.

This run also showed that head 3 attends u at every bundle with an insulin start atom. It skips only the stop-only bundles, where there is no start rate to record. So head 3 is a reliable carrier if the channel is clean.

A clean channel follows from a reference-dimension trick. Put the decoded start rate into one cut dimension jU of the u token, inside the u-encoder. Make dimensions jU and jR exactly zero in the e, slot and z tokens. A head-3 value dimension reading (norm1[jU] − norm1[jR]) is then exactly zero for every token except u. The write becomes (attention to u) × rate, with no gate, using only three state dimensions (jU, jI, jR).

To set this up I need the encoder tensors: the u-encoder's mlp2 and final norm, and the e-encoder's final norm. Next I'll list them and their norm weights on my cheap candidate dimensions.

```python
enc=[n for n in NAMES if any(k in n for k in ['enc','deep','set','mlp1','mlp2','size','z_in'])]
for n in enc: print(n, SHAPES[n])
```