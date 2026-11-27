Confirmed. With the weights the tool actually ran (`WE` without the kv edit), the injection at insulin bundles is a large positive offset in the factual case (+3 to +7.7 in reading units) and does not respond to doubling or halving the dose. That is exactly the upward drift and flat dose response seen in fwd_4.

With the full `WE` (kv edit included):
- the factual injections average near zero (+3.7 to −2.6);
- x2 gives negative shifts (mean −4.9, against −4.2 expected from kappa × rate);
- x1/2 gives a mean of +0.8 against +2.1 expected.

It is noisy per bundle (one x1/2 case has the wrong sign), but on average it is about right.

Fix: save the missing kv tensor so the tool runs the edit I validated. I'll then run a dose_edit on the same 100 tuning stays as fwd_4 to get the real rollout gains. I'll check that the stay list matches fwd_4 and log the finding in the notebook.

```python
import json, numpy as np
save_edit(kvn, WE[kvn].float())
print('edited:', edited())
print('kv saved vs WE max diff %.3e' % np.abs(load(kvn).double().numpy() - to64(WE[kvn])).max())
d4ids = set(pd.read_csv('/work/fwd_4/dose_edit_edited.csv').stay_id.unique())
selA_file = json.load(open('/work/sel_tuning_A.json'))
selA_list = [int(s) for s in (selA_file if isinstance(selA_file, list) else selA_file.get('stays', selA_file))]
print('selA file n', len(selA_list), ' fwd4 ids subset of selA:', d4ids <= set(selA_list), ' sel_A var same:', set(map(int, sel_A)) == set(selA_list))
with open('/work/notes.md','a') as f: f.write("""
- RUN 258-260 ROOT CAUSE: local WE/WCUR contained an UNSAVED rank-2 edit of core.prelude.0.attn.kv.weight (rows 240-287 = v of head 1,
  sv 29462/12075, max 1928) = stage 1 (rate signal from u into slots). Decoders were fit WITH it. Tool ran without it -> factual injection
  +3..+7.7 reading-units at every insulin bundle (drift +12 by h24) and no dose response. With kv (WE): factual inj mean ~0, x2 mean -4.9
  (expected -4.2), x1/2 +0.8 (exp +2.1), noisy per bundle. Run 261: SAVED kv edit -> 5 tensors edited. Next: dose_edit both on selA.
""")
print(json.dumps({"type": "dose_edit", "model": "both", "split": "tuning", "stays": selA_list})[:200])
```