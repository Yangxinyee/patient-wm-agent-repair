The local derivative analysis rules out injecting through z. At insulin and norepinephrine bundles:
- A change of one typical spread in any z direction moves the next predicted glucose or MAP by only about 0.01–0.03 units.
- The directions with no side effects are inconsistent across bundles: their mean effect is smaller than its spread.

So z at drug bundles barely reaches the next prediction, and producing a shift of several mg/dL that way would push z far outside its normal range.

New plan:
- Leave the state update unchanged.
- Read the drugs' history from the state inside the value head, through new gated units in `attr_heads.value_bin`. Insulin units act only on the glucose codes and norepinephrine units only on the MAP codes.
- Fit the linear readout ("probe") by regression on states I generate locally by re-running the stays with randomly rescaled doses. That way dose varies independently of the patient's condition.

This needs spare hidden units in `attr_heads.value_bin.0`, and my encoder's interface so I can re-encode drug atoms at edited rates. The plan is feasible if at least 4 units are inactive on all readings. If none are, I will have to retune a weak unit instead.

```python
def head_hidden(cx, codes, lastv, lastage_s):
    mu, ls = zprior2(cx); view = cx + lin2(mu,'z_head')
    q = Fnn.layer_norm(view + T64('type_emb.weight')[1], (192,), T64('query_norm.weight'), T64('query_norm.bias'), 1e-5)
    cid = torch.tensor([CID[c] for c in codes]); has = torch.tensor(~np.isnan(lastv), dtype=torch.float64)
    zl = torch.tensor([0.0 if np.isnan(v) else float(zsc(c,'value',v)) for c,v in zip(codes,lastv)], dtype=torch.float64)
    ag = torch.tensor(np.where(np.isnan(lastv),0.0,np.log1p(np.nan_to_num(lastage_s))/10.0), dtype=torch.float64)
    ft = torch.zeros((len(codes),8), dtype=torch.float64); ft[:,0]=has; ft[:,1]=ag; ft[:,2]=zl
    lb = torch.tensor([0 if np.isnan(v) else int(fbin(c,'value',np.array([v]))[0]) for c,v in zip(codes,lastv)])
    wc = q + T64('embed.code.weight')[cid] + ft @ T64('last_numeric.weight').T + T64('last_numeric.bias') + T64('embed.bins.value.weight')[lb] + T64('embed.bins.rate.weight')[0] + T64('last_status.weight')[0]
    return lin2(wc,'attr_heads.value_bin.0'), q, wc
cods = list(codes_); lvv = np.array(lv, dtype=float)
preH, qH, wcH = head_hidden(ctxF.double(), cods, lvv, np.asarray(lastage, dtype=float))
mx_ = preH.max(0).values.numpy(); deadH = np.where(mx_ < -3)[0]
print('value_bin.0 units with max pre-act < -3:', len(deadH), '| < -2:', int((mx_<-2).sum()), '| examples', deadH[:12].tolist())
print('enc2 defs:', [l.strip()[:90] for l in open('/work/enc2.py').read().split('\n') if l.strip().startswith('def ')])
```