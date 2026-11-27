That rules out the sampled-z idea: the documentation says "The state update uses the posterior mean."

But line 27 gives a better lead: actions give u_k "**zero if the bundle has none**". My notes say that in states/factual mode the tool feeds the encoder's output for an empty set (u_empty), not zero. My exact local match to fwd_7 confirms that.

If the rollout code path instead uses u = 0 for bundles without drugs, as documented, my edit breaks there. Stage 1 was built to give zero on n1(u_empty), but n1(0) = LN(0) is just the norm1 bias. Head 1 puts 76–95% of its attention on u at reading bundles, so every reading bundle would get a large spurious stage-1 signal. The original model barely reacts to this, which fits its small (0.4) discrepancies.

To test this, I'll rerun the hour-13 predictions locally with u set to zero at bundles without actions, for both models. If the edited model's predictions now match the tool's (for example ≈ 160 for stay 2020000054), the rollout path uses u = 0. The original should still match.

```python
import inspect, numpy as np
sf_ = step_with(WE); print('step fn sig', inspect.signature(sf_))
ue_ = np.asarray(u_empty, float)
def step_u0(W):
    base = step_with(W)
    def f(*args, **kw):
        args = list(args)
        for i_, a_ in enumerate(args):
            if hasattr(a_, 'shape') and tuple(np.shape(a_)) == ue_.shape and np.allclose(np.asarray(a_, float), ue_, atol=1e-9):
                args[i_] = a_ * 0
        return base(*args, **kw)
    return f
rz2 = []
for sid in cmpI.index:
    B = make_bundles(tun[tun.stay_id==sid])
    k = [k for k,(t,ats) in enumerate(B) if 12.0 <= t < 13.0 and any(a.get('kind')==0 and a['code']=='insulin' for a in ats)][0]
    t12, ats = B[k]
    last = [(t, a['value']) for (t, aa) in B[:k+1] for a in aa if a['code']=='glucose_finger' and a.get('value') is not None and a['value']==a['value']]
    tl, xl = last[-1]; dt_s = (13.0-t12)*3600; age_s = (13.0-tl)*3600
    rec = {'sid': sid}
    for nmW, W in [('E', WE), ('O', W64)]:
        r_ = run_tf(B[:k+1], step=step_u0(W)); o_, y_ = r_[-1][0], r_[-1][1]
        rec[nmW+'_u0'] = float(np.asarray(pred_mean(o_, y_, dt_s, 'glucose_finger', xl, age_s, nz=0), float).ravel()[0])
    rz2.append(rec)
RZ2 = pd.DataFrame(rz2).set_index('sid').join(RZ[['E_m0','O_m0','tool_fact_e','tool_fact_o']])
print('E: tool vs u0 median|d| %.2f (vs u_empty %.2f) | O: tool vs u0 %.2f (vs u_empty %.2f)' % ((RZ2.tool_fact_e-RZ2.E_u0).abs().median(), (RZ2.tool_fact_e-RZ2.E_m0).abs().median(), (RZ2.tool_fact_o-RZ2.O_u0).abs().median(), (RZ2.tool_fact_o-RZ2.O_m0).abs().median()))
print(RZ2.round(2).head(8).to_string())
```