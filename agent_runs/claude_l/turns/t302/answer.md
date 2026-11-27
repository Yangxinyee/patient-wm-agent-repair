Raising the gain does nothing. At G = 3, 10 and 30, insulin x2 stays at about −3.2 and norepinephrine at about +0.55, while NLL creeps back to baseline. So the residual injection is being neutralised before it reaches the predictions.

The likeliest cause is the attention split in the prelude. The z token is a near-copy of u, and heads 0–2 share their attention between the two. A larger injection changes the z token's key, so attention shifts back to u and the injected part is ignored. Another possibility is that something in the z_post construction (gate or dose decode) saturates.

Next I'll measure the teacher-forced response directly. I'll double every insulin dose (and separately every norepinephrine dose) in the factual records. Then I'll measure how much the next finger-glucose / arterial-MAP predictions move per unit dose, at lag 1 and lag 2, for G = 0, 1, 10 and 30.

- If the lag-1 slope stays near −0.6 at every G, the injection is lost inside the bundle step, most likely through attention. Then the remedy is to change how the residual enters the prelude, not the gain.
- If the slope does grow with G, the loss happens in the rollout dynamics instead.

```python
t0=time.time()
def dbl(B_, code, f=2.0):
    o=[]
    for b in B_:
        at=[edit_atom(a,f) if (a.get('act') and a['code']==code and a['kind']==0) else a for a in b[1]]
        o.append((b[0],at)+tuple(b[2:]))
    return o
def tf_resp(W, stays, code, c_out):
    X=[];Y=[];L2=[]
    for s in stays:
        B_=BUN[s]; R0=tf_stay_z(W,B_,None); R2=tf_stay_z(W,dbl(B_,code),None)
        dr=drug_recs(B_,code); dmap={int(math.floor(t)):r for t,r in dr}
        for a,b in zip(R0,R2):
            if a[1]!=c_out: continue
            H=int(round(a[0])); dm=b[2]-a[2]
            if (H-1) in dmap: X.append(dmap[H-1]); Y.append(dm)
            elif (H-2) in dmap: L2.append((dmap[H-2],dm))
    X=np.array(X);Y=np.array(Y)
    s1=float((X*Y).sum()/(X*X).sum()); l2=np.array(L2)
    s2=float((l2[:,0]*l2[:,1]).sum()/(l2[:,0]**2).sum()) if len(l2) else float('nan')
    return round(s1,3), round(s2,3), len(X), len(l2)
sub=stays[:50]
for G in [0,1.0,10.0,30.0]:
    Wg=W_cur if G==0 else mkW3(G,G,W_cur)
    print(f'G={G}: insulin->finger slope(lag1,lag2,n1,n2)={tf_resp(Wg,sub,8,CF)} | NE->art={tf_resp(Wg,sub,9,CA)}  t={time.time()-t0:.0f}')
```