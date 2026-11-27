The core replica is exact: relative error 0.0000 on both h and out. The memory entries get the age embedding added before the key/value projection, so both keys and values carry it. Age is in seconds, and the current bundle's vector is already in the memory when the recurrent layers read it.

Together with the exact encoder, I can now compute the model's state locally for any record, including counterfactual dose sequences. That lets me study how a dose change travels through the state, and test candidate weight edits on the factual data before spending forward passes.

Since the conversation can restart at any time, I will first write these verified details into the notebook. In the same run I will check the value-head replica against the tool's factual output (fwd_2). The tool draws z from the prior 4 times, so I compare:
- my replica's expected prediction, averaged over many prior draws;
- the noise floor, measured from my own 4-draw averages.

How to read the result:
- **Head replica exact:** the gap to the tool is about the 4-draw noise floor, and the NLL matches on average.
- **Head replica off:** a gap well above the floor means one of the register inputs, the activation or the time-gap scaling is still off. I would then try the main alternatives.

```python
open('/work/notes.md','a').write("""
## Runs 76-83 (after restart 2): EXACT replicas rebuilt (clean code in kernel: Wo, avec, encset, bundles_of_stay, zpost_of, ctx_of, core_step)
- Bundles: group atoms by round(hour*60). Drug START atoms -> action set u; drug END atoms (no attributes) + measurements + admissions(+END at end_hour) -> e.
- Atom = code+kind+ekind+kq[0]+unit+rate_unit+bins(value,dose,rate)+numeric@[zv,hv,zd,hd,zr,hr]+bias. unit: map 2, glucose_lab 5, finger 0, insulin dose 3, NE dose 4.
  rate_unit: units/hour->2, mcg/kg/min->3 (global rumap in kernel is WRONG/stale). bins: 1+searchsorted(edges,x,'left'). z clipped [-10,10].
- Set enc: LN(mlp2(sum mlp1(a)/sqrt(n) + size(log1p n))); EMPTY set (e or u) -> LN(mlp2(size(0))) (NOT zero). GELU(erf) everywhere.
- z_post exact. CORE STEP EXACT (relerr 0.0000): h'=h*exp(-dt*exp(flow.a)); ctx; z; prelude over [h'+slots, e, u, z_in(z)] (pre-norm: norm1->attn, norm3->ffn), E=prelude_norm(slots);
  blocks 0.0,0.1 on x=h'+E: x+=attn(norm1 x); x+=cross(norm2 x, M+age) where kv=Wkv(M_e + core.age(log1p(age_seconds))) [age added BEFORE kv, both k and v], M includes CURRENT e_k, last 128;
  x+=ffn(norm3 x). hm = abar*h' + softplus(delta_in)*b_in(E) + (x-(h'+E)), abar=exp(-softplus(delta_in)*exp(a_in)); y=c_out(hm); coda: y+=attn(norm1 y); y+=ffn(norm3 y). out=out_norm(mean_slots y).
  MHA: 4 heads x48, no biases, kv rows [0:192]=K,[192:384]=V, scale 1/sqrt(48).
- Dose info enters state ONLY via u_k / z_k tokens in prelude (memory M holds e only; END atoms carry no dose).
""")
import glob
print(glob.glob('/work/fwd_2/*'))
F2 = pd.read_csv(glob.glob('/work/fwd_2/factual_original.csv')[0])
print(F2.shape); print(F2.head(5).to_string()); print(F2.columns.tolist())
```