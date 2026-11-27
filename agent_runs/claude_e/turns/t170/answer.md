The exact cost of the current edit on 100 random tuning stays is +0.0034 nats per reading (1.63461 to 1.63798). By event: glucose_finger +0.0062 per row (about 0.0020 of the total), glucose_lab +0.0013, map_arterial +0.0053 (about 0.0012 of the total), map_cuff 0.

The insulin part is larger than my offline estimate (0.0029 per glucose_finger row). Raising a_ins to about -2.7, as the insulin x2 target requires, would roughly triple it and break the 0.005 budget.

One likely reason for the gap: the tool averages over 4 z-draws, my offline head over 16. Any readout variation that comes from the random z draws is averaged less in the tool and widens the predictive mixture.

Next I'll split each readout's variance into a part across z-draws within a row and a part between rows. A sizeable within-row share would mean I can build cleaner readouts by penalising across-draw variation, which carries no dose information. That would cut the cost without losing rollout response.

```python
def draw_decomp(w, b, msk):
    Rd = np.einsum('dnk,k->dn', WCall[:, msk], np.asarray(w, np.float32)) + b
    within = Rd.var(0).mean(); between = Rd.mean(0).var()
    return within, between
mi = np.isin(code,[6,10]) & (A_.hi.values==1)
for nm, w, b, msk in [('ins ROi5', ROi5['w'], ROi5['w0'], mi), ('ne ROn4', ROn4['w'], ROn4['w0'], tm7), ('ne level', w_neu, b_neu, tm7)]:
    wi, be = draw_decomp(w, b, msk)
    print('%-9s within-draw var %.4f  between-row var %.4f  ratio %.2f  | 4-draw-mean noise var %.4f' % (nm, wi, be, wi/be, wi/4))
dv = np.nan_to_num(A_.dev1.values[mi].astype(float))
print('dev1 var on insulin-glucose rows %.4f ; unit-gain readout implied noise var (between) %.4f' % (dv.var(), draw_decomp(ROi5['w'], ROi5['w0'], mi)[1] - dv.var()))
```