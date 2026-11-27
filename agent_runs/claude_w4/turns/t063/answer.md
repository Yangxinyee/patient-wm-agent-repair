The kernel state confirms that my earlier CPU replica never converged. Several activation and configuration variants were tried with large errors, so I should not trust a guessed replica of the z_post inputs.

A more reliable route is a probe through the forward tool. I temporarily rewrite z_post so its 32 posterior-mean outputs are exact copies of chosen input coordinates. Each copy is passed through a hidden unit with a large positive offset, which works whether the activation is ReLU, GELU or SiLU. Two extra outputs equal act(+0.5) and act(−0.5), which identifies the activation. A "states" request then hands me the true ctx, e_k and u_k values that z_post receives.

Before building that probe I need the exact data format of drug events in the record. A bundle might also hold an END atom of the previous infusion, which would make u_k depend on more than the current rate. So I'm re-reading the data README and a few insulin rows, keeping the output short.

```python
rd_txt = open('/data/README.md').read()
print(len(rd_txt)); print(rd_txt[:2500])
print(tr.columns.tolist())
print(tr[tr.iloc[:,0]==tr.iloc[0,0]].head(12).to_string()[:1800])
```