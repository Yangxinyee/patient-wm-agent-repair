EDIT v4 is saved and behaves as designed in the replica:
- Gating is clean. Lab and cuff rows are unchanged (shift at most 0.06, KL 0), target rows stay at or above 6.5 and non-target rows at or below −7.4.
- Target shifts are centred (finger +0.13 mg/dL, arterial −0.04 mmHg). Spreads are 3.8 and 1.6, with some outliers.
- Direct per-hour head effects have the intended signs: insulin x2 −3.3, x1/2 +1.7; NE x2 +3.2, x1/2 −1.5.

The replica's KL per target row (0.015 finger, 0.020 arterial) suggests a noise-only cost of about 0.01 nats per reading. That is over budget, but the tool, with its z-mixture and real residuals, may differ.

Next is the real factual cost and the readouts from the tool on setA. I need the setA stay ids for the requests, so this run prints them compactly and updates the notebook.

```python
import json
setA = json.load(open('/work/setA.json'))
ids = setA if isinstance(setA, list) else setA.get('stays', setA.get('ids', list(setA.values())[0]))
ids = [int(x) for x in ids]
print(len(ids)); print(','.join(str(x) for x in ids))
with open('/work/notes.md', 'a') as f:
    f.write('- Run 273-277: LDA gates (z-orthogonal, finger vs rest lamf .01, art vs rest lamf .03) margin/sd 36/45. Dead units [114,10,0,107] (ins A,B; NE A,B).\n'
            '  Decoders DECR2: orthogonal to z-subspace + last_numeric + bins.value/rate + last_status + code emb (85 dims), kap .3: ins slope .522 per-unit noise 1.53; NE slope .683 per-unit .279.\n'
            '  build_v4(gI,gN,...) 2-unit difference, k=4/p99.99, tilt along centred centres /medvar, Vp-weighted exact cell centring via TR. SAVED gI=-2.0 gN=2.5:\n'
            '  replica: lab/cuff untouched; finger shift mean .13 sd 3.8 KL .0153; art mean -.04 sd 1.6 KL .0203; direct/h ins2 -3.26 ins.5 1.72 ne2 3.17 ne.5 -1.46.\n'
            '  Earlier failures: v1 big-bias gate (tool NLL 16-18); v3 decoder not code-invariant -> leak. NEXT: tool factual both + dose_edit both on setA.\n')
```