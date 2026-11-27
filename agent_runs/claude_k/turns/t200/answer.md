The inventory shows the kernel still holds most of the reconstruction machinery:
- `view_from`, `ctx_from`, `zmean`, `setenc_my`, `datom`, `u_rec`;
- q, with_code and value-head arrays for 99 stays;
- dose-innovation arrays for all 30,000 stays;
- the insulin and norepinephrine decoders and U_EMPTY;
- the fwd8 states in both versions;
- many unlabelled arrays from runs 168–183 (CTX, ZP_prior, Qo, Qe, dQ, Z18, Z28, META and others).

Since those later arrays are undocumented, I will rebuild anything I depend on from named functions rather than trust them.

The decisive design question now is how a write into a quiet z dimension at a drug bundle reaches q at later readings. That decides whether one linear readout in the value head can produce the right lag kernel. It also decides whether the expected-dose compensation belongs in the writer, decoded from ctx, or elsewhere. Both depend on how the recurrent state update uses the z posterior mean, and on what ctx is.

My notes record only fragments of this. So I will re-read ARCHITECTURE.md, focusing on the state update, the z prior and posterior, and the view. If z enters the state additively or through a gated slot write, the lag response can be predicted and a readout fitted from one `states both` pass. If it enters through strongly nonlinear paths, I will need empirical calibration with more forward passes.

```python
t = open('/model/ARCHITECTURE.md').read()
print(len(t))
print(t[:9800])
```