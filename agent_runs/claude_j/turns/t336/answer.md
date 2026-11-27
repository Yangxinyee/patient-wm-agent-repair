The step equations show that u_k enters the state only as a token in the prelude attention, where head 2 attends to it on drug bundles. The z route is unused. So the model's existing, too‑weak dose response must pass through that head.

The simplest test is to scale head 2's value rows by k=4 and measure with the tool:
- if the four shifts grow about fourfold while the factual cost stays small, I calibrate k;
- if they shrink or flip, this head carries the opposite‑sign part and k<1 is the direction to try;
- if the factual cost is large, the approach is ruled out.

```python
kv=K0.copy(); kv[288:336]*=4.0
save_edit(KV,TT(kv)); print('edited',edited(), float(np.abs(A(load(KV))-K0).max()))
with open('/work/notes.md','a') as f:
    f.write("- fwd7 (75 sel1 stays): channel mk(3,-3) ~no effect (A x2 -2.65->-2.57, B x2 .745->.787). Channel abandoned, dropped.\n- Arch: u_k enters state only as prelude token (z path unused). Test: scale prelude head-2 value rows 288:336 by k=4.\n")
```