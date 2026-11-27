Run 189 settles three points.

**Action-only reads are clean.** Fitted only at action bundles, head2's mixture reads the start rates well: 0.055 u/h for insulin and 0.038 for norepinephrine. On reading bundles such a read explodes (sd ~1500), so it needs a gate.

**Gating is barely possible.** A linear "action present" read separates the two bundle types, but with a margin of only 0.25. That makes a gated channel feasible but fiddly.

**Dead units exist.** The prelude FFN has 154 dead hidden units (pre-activation always below −2).

Weighing this: gating would only cut the read noise from about 0.36 to 0.14 u/h (insulin) and from 0.21 to 0.10 (norepinephrine). The ungated noise costs an estimated ~0.0004 (glucose) to ~0.003 (MAP) nats against the 0.005 limit. So I'll build the simpler ungated channel first and add gating only if the factual fit demands it.

Next I'll note this down. Then I'll look at the state dynamics that decide how long an injected component persists:
- per-dimension a_bar and time-flow timescales;
- how diagonal c_out is;
- the per-dimension norm-weight scales.

If some dimensions forget within the hour and others keep about 0.35–0.5 per hour, I can tune the lag-2 term through persistence instead of building a separate END-atom channel.

```python
open('/work/notes.md','a').write("""
- Runs 177-189: LN outputs small (E ~13/slot, out ~14); residual streams huge (x2 ~640/slot). Natural low-variance dirs exist at every stage
  (x2 min sd .024, mean-y .005, E .0006, out .0001) -> small injections along them are readable.
- Prelude head2: att on u .989 at action bundles, but on e .92 at no-action. Linear read of rate from n1 tokens: clean on u only (sd .05),
  but conflicts with e tokens: best joint (attention-mixture) read resid sd ins_s .36 (no-action .20), ne_s .21 (.07). Action-only fit .055/.038
  but garbage sd ~1500 at no-action. Presence read margin small (act min -.58 vs noact max -.83). Prelude ffn has 154 dead units (gating possible).
  DECISION: first build UNGATED channel (noise est. ~0.0004-0.003 nats), add gating only if factual fit needs it.
""")
fa_=Wt['core.flow.a'].numpy().ravel(); ain=Wt['core.a_in'].numpy().ravel(); din=Wt['core.delta_in'].numpy().ravel()
sp_d=np.log1p(np.exp(din)); abar_=np.exp(-sp_d*np.exp(ain)); tau=1/np.exp(fa_)/60
print('flow tau minutes: quantiles', np.round(np.quantile(tau,[0,.05,.1,.25,.5,.75,.9,1]),1))
print('a_bar quantiles', np.round(np.quantile(abar_,[0,.1,.25,.5,.75,.9,1]),3), ' sp quantiles', np.round(np.quantile(sp_d,[0,.25,.5,.75,1]),3))
hr_keep=abar_*np.exp(-300/60/tau)*abar_*np.exp(-3300/60/tau)   # two steps (5min then 55min) per hour, diag only
print('diag per-hour keep (2 steps) quantiles', np.round(np.quantile(hr_keep,[0,.1,.25,.5,.75,.9,1]),3))
Co=Wt['core.c_out.weight'].numpy(); print('c_out shape',Co.shape,'diag frac', np.round(np.abs(np.diag(Co)).sum()/np.abs(Co).sum(),3), 'sv range', np.round(np.linalg.svd(Co,compute_uv=False)[[0,-1]],3))
print('out_norm w quantiles', np.round(np.quantile(np.abs(Wt['core.out_norm.weight'].numpy()),[0,.1,.5,.9,1]),3),' prelude_norm w', np.round(np.quantile(np.abs(Wt['core.prelude_norm.weight'].numpy()),[0,.1,.5,.9,1]),3))
```