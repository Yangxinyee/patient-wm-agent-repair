The NLL gap is almost entirely in the codes other than finger glucose:

| code | my NLL | tool NLL |
|---|---|---|
| finger glucose | 1.520 | 1.526 |
| lab glucose | 1.272 | 1.214 |
| arterial MAP | 1.381 | 1.264 |
| cuff MAP | 1.664 | 1.555 |

So the head is right for the main readout code, finger glucose. The other three get information I'm not giving them.

The likely source is the other readings in the same bundle. If atoms within a bundle are predicted one after another, a cuff MAP would be predicted knowing the arterial MAP at the same minute, and a lab glucose knowing the finger glucose.

Test: break down the per-reading NLL gap (mine minus tool) by whether a partner reading is in the same bundle and by whether NE was running. A large gap only when a partner is present would confirm within-bundle conditioning.

```python
fa4['nl_mine'] = nl_; fa4['dnl'] = fa4.nl_mine - fa4.nll
part = {'glucose_finger':'glucose_lab','glucose_lab':'glucose_finger','map_arterial':'map_cuff','map_cuff':'map_arterial'}
bset = fa4.groupby(['stay_id','hour']).event.apply(set).to_dict()
fa4['partner_same'] = [part[e_] in bset[(s_, h_)] for s_, h_, e_ in zip(fa4.stay_id, fa4.hour, fa4.event)]
fa4['n_b'] = [len(bset[(s_, h_)]) for s_, h_ in zip(fa4.stay_id, fa4.hour)]
fa4['prev_drug'] = (S['has_insulin'][fa4.prev.values] + S['has_norepinephrine'][fa4.prev.values]) > 0
fa4['ne_prev'] = S['has_norepinephrine'][fa4.prev.values] > 0
for ev in ['map_arterial','map_cuff','glucose_lab','glucose_finger']:
    d_ = fa4[fa4.event==ev]
    print(ev, d_.groupby('partner_same').dnl.agg(['mean','size']).round(3).to_dict(), '| ne_prev', d_.groupby('ne_prev').dnl.mean().round(3).to_dict(), '| n_b', d_.groupby('n_b').dnl.mean().round(3).to_dict())
```