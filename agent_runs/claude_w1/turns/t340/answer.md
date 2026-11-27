The test supports my explanation of the v4 failure. The v4 decoder term k·v·x carried a constant of about 145 on every reading code (|k·v| ≈ 305–318), while the dose signal was only 0.64–0.70 on top of it.

Since that term reads q, which is LayerNorm'd, a few-percent change in the LayerNorm scale from z draws moves the constant by about 5–7 units. That is enough to switch gated units on for lab and cuff rows.

Next run builds v5 decoders, one per drug:
- **Unit response:** minimum variance (MVDR) subject to a unit response per lag-1 dose change. The response comes from the replica edit responses EDR, and the noise from within-cell residuals of the target code's rows.
- **Constraints:** each decoder is orthogonal to the gamma-weighted, centred z_head directions (z-draw noise), to all register/code embedding tensors (deterministic features), and to the per-code mean head input. The last constraint removes the constant offset.

I'll then rebuild with build_v4 at the old gains and check that:
- the decoder's mean output per code is about 0;
- the units' pre-activations keep clean gate margins.

Small offsets and clean margins would support the v5 design; the tool check comes next.

```python
import io, contextlib
gq=np.asarray(load_original('query_norm.weight'))
Wz=np.asarray(load_original('z_head.weight')); Wz=Wz if Wz.shape[0]==192 else Wz.T
Cc=np.eye(192)-np.ones((192,192))/192; Zq=gq[:,None]*(Cc@Wz)
cols=[Zq]
nn=[n for n in NAMES if any(s in n for s in ['last_numeric','last_status','bins.value','bins.rate','embed.code'])]
print('nuis', [(n,tuple(SHAPES[n])) for n in nn])
for n in nn:
    t=np.asarray(load_original(n))
    if t.ndim==1 and t.shape[0]==192: cols.append(t[:,None])
    elif t.ndim==2 and t.shape[1]==192: cols.append(t.T)
    elif t.ndim==2 and t.shape[0]==192: cols.append(t)
Xa=np.asarray(Xf); Ra=np.asarray(Rf)
cols.append(np.stack([Xa[cfa==c].mean(0) for c in range(4)],1))
Nm=np.concatenate(cols,1); U_,S_,_=np.linalg.svd(Nm,full_matrices=False); rk=int((S_>1e-6*S_[0]).sum()); Ub=U_[:,:rk]
Q=np.linalg.svd(np.eye(192)-Ub@Ub.T)[0][:,:192-rk]; print('nuisance rank', rk, 'free', Q.shape[1])
DEC5={}
for drug,code in [('insulin',0),('norepinephrine',2)]:
    ok=[EDR[(drug,f)]['reached'].astype(bool) for f in (2.0,0.5)]
    dX=np.concatenate([EDR[(drug,f)]['dX'][m] for f,m in zip((2.0,0.5),ok)]); d1=np.concatenate([EDR[(drug,f)]['dd'][m,0] for f,m in zip((2.0,0.5),ok)])
    r=(dX*d1[:,None]).sum(0)/(d1**2).sum()
    R=Ra[cfa==code]; Sg=Q.T@(R.T@R/len(R))@Q; Sg+=1e-4*np.trace(Sg)/len(Sg)*np.eye(len(Sg))
    rq=Q.T@r; a=np.linalg.solve(Sg,rq); a/=rq@a; v=Q@a; pr=dX@v; sl=(pr*d1).sum()/(d1**2).sum()
    print(drug,'noise/unit %.3f'%np.sqrt(a@Sg@a),'slope %.3f'%sl,'resp scat %.3f'%(pr-sl*d1).std(),'|v| %.1f'%np.linalg.norm(v),
          'mean/code',[round(float((Xa[cfa==c]@v).mean()),3) for c in range(4)],'zsens %.1e'%np.abs(v@Zq).max())
    DEC5[drug]=v
buf=io.StringIO()
with contextlib.redirect_stdout(buf): build_v4(-2.0, 2.5, DEC5['insulin'], DEC5['norepinephrine'], 1.0, 1.0)
print('build:', buf.getvalue()[-700:])
W1n=np.asarray(load('attr_heads.value_bin.0.weight')); b1n=np.asarray(load('attr_heads.value_bin.0.bias'))
P=Xa@W1n[U].T+b1n[U]
print('norms',[round(float(np.linalg.norm(W1n[u])),1) for u in U],'b',[round(float(b1n[u]),1) for u in U])
for c in range(4): print('code',c,np.round(P[cfa==c].mean(0),1),np.round(P[cfa==c].std(0),2))
kv5=W1n[114]-W1n[10]; kn5=W1n[0]-W1n[107]
print('kvx mean/code ins',[round(float((Xa[cfa==c]@kv5).mean()),2) for c in range(4)],'ne',[round(float((Xa[cfa==c]@kn5).mean()),2) for c in range(4)], 'edited', edited())
```