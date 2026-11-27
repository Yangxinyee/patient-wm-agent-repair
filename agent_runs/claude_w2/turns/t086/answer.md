Run 80 shows the linear surrogate cannot stand in for the model's rollout. The early lags roughly match the tool's original-model profile:

| Lag | Surrogate | Tool |
|---|---|---|
| 1 | −0.47 | −0.55 |
| 2 | −0.81 | −0.87 |
| 3 | −1.6 | −1.1 |
| 4 | −2.3 | −1.5 |

After that the surrogate swings back to 0 by lag 11–12. The tool keeps accumulating almost linearly, to −3.5 at lag 12. Likewise, the records' AR(3) version grows without bound (−27.9 at lag 12), unlike the concave K-based truth (about −19 at lag 12).

So a 3-lag linear projection misreads long-horizon behaviour. The positive late-dose coefficients are probably partly stand-ins for glucose lags I left out. In the real rollout the model carries a drawn glucose shift almost fully from hour to hour.

(Also, `r` holds norepinephrine rows: truth +4.75, model +1.08. I still need the per-lag insulin truth.)

**What this means for the edit.** The truth is concave: the incremental effect of a sustained dose shrinks over about 10 h. With near-unit propagation, the correction would need a large lag-1 term (≈ −2/U/h) plus a positive tail over lags 3–10. That needs a memory of past doses, and the state holds the dose deviation only weakly (R² ≈ 0.46 for the current one). I cannot check offline how a state-based decoder responds to doubled doses.

**Decision.** I will build a local re-implementation of the full forward pass. It gives three things:
- states under counterfactual doses;
- local rollouts;
- local factual NLL checks.

I can check it step by step against the fwd_3 states (out, h, z per bundle).

Before writing new code I record the session's findings in the notebook, and test whether my earlier helpers (`stay_atoms`, `atom_vec`, `encode`) still work and what they return.

```python
import inspect
notes_new = """
## Restart 3 (runs 72-81)
- ARCH re-read: head 'last' register = last value of SAME code (no dose in head). Head z drawn from PRIOR 4x, probs averaged (explains my head NLL gap, run 71).
  core: a_bar 0.31-0.46 per bundle; h rewritten each step by prelude/blocks/coda (pre-norm transformer layers, 4 heads, ff 768); no dead h dims. => no cheap dedicated memory channel.
- Bins: gf centres 72..271 (top bin >=260 -> centre 271), ma 33..101 (>=98). insulin rate/dose bins saturate >5; NE rate bin 16 for >1.0.
- One-step OLS gf_t on gf lags1-3, ins lags1-10, pol lags1-4 (train n=228k): AR .48/.30/.21 (sum .98); dose -1.99 -1.47 -0.89 +.17 +.81 +.65 +.40 +.32 +.23 +.03.
  Model predicted_mean (fwd_4 selT, n=1704, resid sd 3.2): AR .46/.24/.17 (sum .87, const 18); dose -0.31 -0.10 -0.49 -0.04 +.51 +.31 +.32 +.31 +.05 +.10.
  => model misses short-lag insulin effect (lags1-3), compensates with stronger mean reversion.
- Linear surrogate rollout (sel1, x2): model-coef surrogate -1.2 (tool -2.14), but shape wrong at late lags (surrogate rebounds, tool grows ~linearly -0.55 -> -3.53).
  => linear AR(3) projection is NOT a reliable rollout model. Model propagates drawn shifts ~fully.
- DECISION: build local full re-implementation (core step verified vs fwd_3 states), then local rollouts & counterfactual states.
"""
with open('/work/notes.md', 'a') as f_: f_.write(notes_new)
for fn_ in ['stay_atoms', 'atom_vec', 'atom_vec2', 'encode', 'ctx_from', 'zpred', 'ctx_v', 'gapv', 'LN_', 'ln', 'mlp', 'lin', 'zsc', 'fbin']:
    try: print(fn_, inspect.signature(globals()[fn_]), globals()[fn_].__code__.co_varnames[:12])
    except Exception as ex: print(fn_, 'ERR', ex)
sid_ = selT[0]
try:
    A_ = stay_atoms(sid_)
    print(type(A_)); print(A_[:3] if isinstance(A_, list) else (A_.head(8).to_string() if hasattr(A_, 'head') else A_))
except Exception as ex:
    print('stay_atoms failed:', ex)
print(trs.columns.tolist(), trs.head(3).to_string())
```