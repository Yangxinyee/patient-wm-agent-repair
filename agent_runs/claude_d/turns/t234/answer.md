Norepinephrine dose information is weak even in the full state, not just in the head's input:
- For the rate deviation, all 992 state numbers give R² 0.30, against 0.17 from with_code.
- For the one-step surprise they give 0.20, against 0.11.
- For the rate itself they give 0.57, against 0.46, and much of that comes from MAP history.

So the model barely represents norepinephrine doses, and no edit of the readout can recover a clean dose signal.

An unbiased norepinephrine channel would cost roughly 0.002–0.009 nats, depending on its strength. Its rollout response is uncertain. Norepinephrine therefore waits until the insulin edit is final and its true factual cost on the GPU is known. That cost also needs checking because offline the retired units appeared to improve MAP NLL by 0.0036, which seems suspicious.

Now the calibrated insulin edit. Scaling C by 0.84 gives C = −2.79. This should bring insulin x2 to about −13.8 and x1/2 to about +6.9 on these stays. It should also lower the factual cost, because C moves closer to the lag-1 shortfall of −1.6.

I will build it, check it offline in fp32 and bf16, save it, and then measure its factual NLL on the GPU.

```python
E3b = build_ins3(-2.79, ws_f, bs_f, k_f, wfp_f)
nll_e, mean_e = eval_E(*E3b); _, mean_e16 = eval_E(*E3b, dt=torch.bfloat16)
d_ = nll_e - nll_o
print('E3b in-sample dNLL all %.5f | glu&pres %.4f glu&abs %.4f map %.4f | test-stays %.5f' % (d_.mean(), d_[mk].mean(), d_[glu&~pres_i].mean(), d_[mapf].mean(), d_[tem].mean()))
print('bf16-fp32 bias glu&pres %.3f' % (mean_e16-mean_e)[mk].mean())
for nm_, T_ in zip(['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias'], E3b):
    save_edit(nm_, torch.tensor(T_))
np.savez('/work/E3b.npz', *E3b)
with open('/work/notes.md','a') as f:
    f.write("- Run 210: NE info weak even in full state (R2 D_ne_1 full .30, wc .17; eps_ne_1 .20/.11; rate .57/.46) -> NE head channel limited. E3b = E3-ins with C=-2.79 saved (/work/E3b.npz).\n")
print(edited())
```