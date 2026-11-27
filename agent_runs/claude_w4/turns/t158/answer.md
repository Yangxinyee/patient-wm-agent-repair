Removing the cross-attention barely changes the error (207 without, 220 with), and the error grows with position in the stay. So the memory isn't the main problem. Something in the prelude-to-block path or the update is off.

Diagnostic: regress the true change (GPU y − h_tf) on my replica's change, element-wise:
- slope near 1 with moderate fit → a missing additive piece;
- slope well below 1 or a poor fit → a deeper error, such as the wrong E or a wrong update term.

I'll also compare against two simpler predictors: the h_tf decay alone, and the coda applied to c_out(h_tf).

```python
def fitstat(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    sl = (a*b).sum()/(b*b).sum(); r2 = 1 - ((a - sl*b)**2).sum()/((a - a.mean())**2).sum()
    return round(sl.item(),3), round(r2.item(),3)
print('d(y) vs d(rep)', fitstat(yt-htf, y1-htf), 'no cross', fitstat(yt-htf, y2-htf))
def coda_only(hn):
    c = hn @ Wc['core.c_out.weight'].T
    n1 = LNw(c, 'core.coda.0.norm1'); c = c + mha_m(n1, n1, Wc['core.coda.0.attn.q.weight'], Wc['core.coda.0.attn.kv.weight'], Wc['core.coda.0.attn.out.weight'])
    return c + ffn(LNw(c, 'core.coda.0.norm3'), 'core.coda.0.')
y3 = coda_only(htf)
print('coda(htf) err', round((y3-yt).norm(dim=(1,2)).mean().item(),1), 'htf err', round((htf-yt).norm(dim=(1,2)).mean().item(),1))
print('c_out near identity?', round((Wc['core.c_out.weight'] - torch.eye(192)).norm().item(),2), round(Wc['core.c_out.weight'].norm().item(),2))
```