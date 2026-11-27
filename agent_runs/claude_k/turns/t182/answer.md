The corrected rebuild of u matches the tool on all 3305 drug bundles:
- **Insulin projection:** error 0.047, which is about 1e-4 in u along p̂.
- **Norepinephrine projection:** error 0.012.
- Both-drug and norepinephrine-only bundles are as good as single-drug ones.

So u is built from start atoms only. Each carries categorical rows: insulin unit 3, rate_unit 2, route 1; norepinephrine unit 4, rate_unit 3, route 1. The empty case is U_EMPTY.

I'll log this, then refit both dose decoders on exact encodings, covering:
- all recorded (insulin, norepinephrine) rate pairs;
- each drug multiplied by 0 (atom kept at rate 0), ½, 1.5, 2 and 3;
- the drug removed altogether;
- the empty set, constrained to read 0.

Insulin targets are capped at 11.5, because the input z-score clips there. I'll also check the z_in weights and typical token size, to pick a writer gain that keeps the z shift moderate.

What I'm looking for: decoder errors ≲0.05 units/h (insulin) and ≲0.01 (norepinephrine) across all edits, with a modest decoder norm. If the edited rates can't be fitted linearly, I'll need several units instead of one.

```python
with open('/work/notes.md','a') as fh:
    fh.write("- run150-167: WRITER BUG FOUND. (1) empty-action u = U_EMPTY (not 0): p_ins.U_EMPTY=-19.2, p_ne.U_EMPTY=-0.51 -> fwd8 had const dz18=-1.5, dz28=-1.02 on all non-drug bundles.\n"
             "  (2) TRUE u = encode_action over START atoms only (END atoms excluded although n_atoms counts them); drug atoms include categorical rows: insulin unit 3, rate_unit 2, route 1; NE unit 4, rate_unit 3, route 1; action 0, status 0. bins = 1+#edges<x (strict). z clip +-10 (insulin rate>11.5 saturates).\n"
             "  corrected recon u_rec(ri,rn) matches tool-implied projections: rmseI .047 (|p|379), rmseN .012. Functions datom(), u_rec(), u_c() in kernel.\n")
pairs=pd.DataFrame(dict(a=ins_all.ravel(), b=ne_all.ravel())); pairs=pairs[(pairs.a>0)|(pairs.b>0)]
cnt=pairs.groupby(['a','b']).size().reset_index(name='w'); print('distinct factual pairs',len(cnt))
Ulist=[];YI=[];YN=[];Wt=[]
def add(u_,yi,yn,w): Ulist.append(u_); YI.append(yi); YN.append(yn); Wt.append(w)
for a,b,w in cnt.itertuples(index=False):
    for m in [0,0.5,1,1.5,2,3]:
        if a>0: add(u_c(a,b,mi=m), min(a*m,11.5), b, w)
        if b>0 and m!=1: add(u_c(a,b,mn=m), min(a,11.5), b*m, w)
    if a>0 and b>0:
        add(u_c(0,b), 0, b, w); add(u_c(a,0), min(a,11.5), 0, w)
add(U_EMPTY.copy(),0,0,cnt.w.sum())
U_=np.array(Ulist); YI=np.array(YI); YN=np.array(YN); Wt=np.array(Wt,float); Wt=Wt/Wt.mean()
print('training rows',len(U_))
def wridge(X,y,w,al):
    Xw=X*np.sqrt(w)[:,None]; yw=y*np.sqrt(w)
    return np.linalg.solve(Xw.T@Xw+al*np.eye(X.shape[1]), Xw.T@yw)
for al in [1e-4,1e-3,1e-2,1e-1]:
    pi_=wridge(U_,YI,Wt,al); pn_=wridge(U_,YN,Wt,al)
    ei=U_@pi_-YI; en=U_@pn_-YN
    print('alpha',al,'|pI|',round(np.linalg.norm(pi_),1),'rmseI',round(np.sqrt(np.average(ei**2,weights=Wt)),4),'maxI',round(np.abs(ei).max(),3),'| |pN|',round(np.linalg.norm(pn_),1),'rmseN',round(np.sqrt(np.average(en**2,weights=Wt)),4),'maxN',round(np.abs(en).max(),3),'empty',round(float(U_EMPTY@pi_),3),round(float(U_EMPTY@pn_),4))
print('zin col norms 18,28', np.linalg.norm(zin_W[:,18]), np.linalg.norm(zin_W[:,28]), 'median col norm', np.median(np.linalg.norm(zin_W,axis=0)))
```