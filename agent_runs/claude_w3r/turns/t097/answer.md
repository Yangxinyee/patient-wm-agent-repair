Run 90 found 17 hidden units in the value head that are dead on every value row (max activation < 0.05) and have no effect on the output. I can repurpose them without changing anything else.

That allows a clean gated channel built from two dead units:
- Unit A gets gate + probe, unit B gets the gate alone, and the output is u·(GELU(A) − GELU(B)).
- On glucose_finger rows both units sit in GELU's linear range, so the gate cancels and the output is u·probe.
- On every other code both are deep negative, so the output is exactly 0.
- The same construction with two more units handles norepinephrine → map_arterial.

The probe will be a linear readout of `with_code`, fitted to the insulin deviation. It will be rescaled by the inverse of its slope on the true deviation (about 1/0.41). Its coefficient will be the records' effect minus the model's own response.

This step: record the plan in my notes, recompute `with_code` for all value rows (stored for gate and probe fitting), and inspect panel.pkl to get the policy cell means.

```python
with open('/work/notes.md','a') as f:
    f.write("""
- Run 83-88: local head check. Register age must be measured at STATE time (minutes): log(1+ (t_state - t_last)*60)/10. typ=1, num[3]=1, last_status unused.
  Remaining error median ~2 mg/dL (ratio 1.4 vs 4-draw noise) -> local head approximately right, not exact. Use end-to-end calibration.
- Run 89 (gf rows of fwd_3, train selS): R2 from q: rate 0.69, dev(=r-bandmean) 0.41. MODEL partial slope of pred on dev = +0.31 (wrong sign: dose used as glucose proxy).
  Two-feature view: state holds clean band + noisy dose f=r+e (sd_e~0.9). Head term c*(f-m) with c=b1-gamma=-2.31 gives correct interventional slope,
  factual cost only ~+0.0004 nats/gf reading. Probe p=E[dev|q] has slope k~0.41 on true dev -> use T=(c/k)*p.
- Policy: insulin never in hours<=2; after that P(r=0|g>=155)=0.08.
- Run 90: value head (Linear192-GELU-Linear17, input with_code) has 17 DEAD hidden units (max act<0.05 over all 5684 rows).
  PLAN: gated linear channel with 2 dead units: A=GELU(G+P), B=GELU(G), W2[:,A]=u, W2[:,B]=-u -> adds u*P on gf rows (G>>0), 0 elsewhere (G<<0).
  u = tilt vector (c_b - cbar)/Vbar (u_0=0). Same for norepi->map_arterial with 2 more units.
""")
XW = np.zeros((len(MT),192), np.float32); QW = np.zeros_like(XW)
for n, r_ in enumerate(MT.itertuples()):
    w_, hr = bidx[r_.stay_id]; i0 = r_.i0
    out_ = torch.tensor(SX[i0,:192]); h_ = torch.tensor(SX[i0,192:960]).reshape(4,192)
    dt = (r_.hour - SHR[i0])*3600.0; ctx = HD.ctx(out_, h_, dt); mu, ls = HD.zprior(ctx)
    q = HD.LN(ctx + HD.L(mu,'z_head') + W['type_emb.weight'][1],'query_norm')
    ph, pv = LASTREG.get((r_.event, r_.stay_id, round(r_.hour,4)), (np.nan,np.nan)); num = torch.zeros(8); lb = 0
    if pv==pv:
        lb = fbin(r_.event,'value',float(pv)); num[0]=1; num[1]=math.log(1+max(SHR[i0]-ph,0)*60)/10; num[2]=zsc(r_.event,'value',float(pv)); num[3]=1
    XW[n] = (q + W['embed.code.weight'][CODE[r_.event]] + HD.L(num,'last_numeric') + W['embed.bins.value.weight'][lb]).numpy(); QW[n] = q.numpy()
chk = np.abs(Fnn.gelu(torch.tensor(XW) @ W['attr_heads.value_bin.0.weight'].T + W['attr_heads.value_bin.0.bias']).numpy() - FA).max()
print('XW recomputed; max diff vs FA', chk)
PN = pd.read_pickle('/work/panel.pkl'); print(type(PN), getattr(PN,'shape',None)); print(PN.columns.tolist() if hasattr(PN,'columns') else list(PN)[:10]); print(PN.head(3).to_string() if hasattr(PN,'head') else '')
```