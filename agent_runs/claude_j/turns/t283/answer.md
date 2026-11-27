Gating on a preceding drug bundle would not help. About 85–88% of readings follow a drug bundle, and those rows carry almost all of both the cost and the cf effect. I now know the head layout:
- value_bin.0 has 192 hidden units and value_bin.2 maps them to 17 logits.
- embed_transition has 13 codes × 17 last-bins.
- Codes: glucose_finger 6, glucose_lab 10, map_arterial 7, map_cuff 5, insulin 8, norepinephrine 9.

Plan: build the edit and test it with the tool, since the rollout response is the real unknown.
- **Units:** unit 89 (insulin → glucose codes) and unit 170 (drug B → MAP codes). Each reads a full-data ridge readout, kept orthogonal to the last_numeric columns.
- **Gate:** a minimum-variance code gate, about +40 for its own codes and −40 for all others, plus an offset that keeps the unit in GELU's linear range.
- **Output and centring:** the output column is the tilt vector. The factual mean of each (code, last bin) group is subtracted in embed_transition.

I start at gain 1.5 for both, then check the gate margins.

```python
def L_(n):
    t=load(n); return (t.detach().cpu().numpy() if hasattr(t,'detach') else np.asarray(t)).astype(np.float64)
bn=[n for n in NAMES if n.startswith('bins')]; print('bins',bn[:4])
EC=L_('embed.code.weight'); LNW=L_('last_numeric.weight'); BV=L_(bn[0]) if bn else None
Pn=np.eye(192)-LNW@np.linalg.pinv(LNW)
def fit_full(fa_,c_,v_,rho,lam):
    c17c=np.stack([cenof(e) for e in c_.event])[:,None,:]; T=(c_.S-rho*c_.Sp-(c_.mean_cf-c_.mean_f)).values
    th=np.clip(tsolve(LG[c_.ri.values],v_,c17c,T),-1.5,1.5); Xc_=(WC[c_.ri.values]-WC[c_.fi.values])@Pn
    Xf_=WC[fa_.ri.values]; gk=(fa_.event+'_'+fa_.lb.astype(str)).values
    Xft=(Xf_-pd.DataFrame(Xf_).groupby(gk).transform('mean').values)@Pn
    A=Xc_.T@Xc_+lam*Xft.T@Xft; A+=1e-2*np.trace(A)/192*np.eye(192); return Pn@np.linalg.solve(A,Xc_.T@th), Xft
facall=Mi[Mi.drug=='none']; Sig=np.cov(WC[facall.ri.values].T)+1e-3*np.eye(192); Si=np.linalg.inv(Sig)
def gate(pos,G=40.):
    t=np.array([G if c in pos else -G for c in range(13)]+[0.0]*8); Cm=np.vstack([EC,LNW.T]); return Si@Cm.T@np.linalg.solve(Cm@Si@Cm.T,t)
cid_of={'glucose_finger':6,'glucose_lab':10,'map_arterial':7,'map_cuff':5}
rowsM=Mi[Mi.event.isin(list(cid_of))]; cidM=rowsM.event.map(cid_of).values
def build_unit(w,Xft,pos,fa_,v_,g):
    kap=1.5/np.std(Xft@w); u=kap*w+gate(pos); s=WC[rowsM.ri.values]@u; ip=np.isin(cidM,pos); B=5.0-s[ip].min()
    a_pos=s[ip]+B; a_neg=s[~ip]+B; xr=WC[rowsM.ri.values]-EC[cidM]; ur=xr@u
    oth=[c for c in range(13) if c not in pos]; drugmax=max(EC[c]@u for c in oth)+ur.max()+B
    af=WC[fa_.ri.values]@u+B; cc=fa_.event.map(cid_of).values; mu=np.zeros((13,17))
    for c in pos:
        mk=cc==c; base=af[mk].mean()
        bvm=(BV[fa_.lb.values[mk]]@u).mean() if BV is not None else 0
        for lb in range(17):
            m2=mk&(fa_.lb.values==lb)
            mu[c,lb]=af[m2].mean() if m2.sum()>=5 else base+((BV[lb]@u-bvm) if BV is not None else 0)
    print('kap',round(kap,1),'B',round(B,1),'a_pos min/med',round(a_pos.min(),1),round(np.median(a_pos),1),'a_neg max',round(a_neg.max(),1),'other-code max est',round(drugmax,1))
    return dict(u=u,B=B,col=(g/kap)*v_,mu=mu,pos=pos,kap=kap)
w_ins,Xft_i=fit_full(fac,cf,vG,0.4,30); w_ne,Xft_n=fit_full(faN,cN,vM,0.3,300)
U_ins=build_unit(w_ins,Xft_i,[6,10],fac,vG,1.5); U_ne=build_unit(w_ne,Xft_n,[7,5],faN,vM,1.5)
def apply_units(units):
    H0=L_('attr_heads.value_bin.0.weight'); b0=L_('attr_heads.value_bin.0.bias'); H2=L_('attr_heads.value_bin.2.weight'); ET=L_('embed_transition.value_bin.weight')
    for j,U in units:
        H0[j]=U['u']; b0[j]=U['B']; H2[:,j]=U['col']
        for c in U['pos']:
            for lb in range(17): ET[c*17+lb]-=U['col']*U['mu'][c,lb]
    for n,a in [('attr_heads.value_bin.0.weight',H0),('attr_heads.value_bin.0.bias',b0),('attr_heads.value_bin.2.weight',H2),('embed_transition.value_bin.weight',ET)]:
        save_edit(n,torch.tensor(a,dtype=torch.float32))
for n in ['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','embed_transition.value_bin.weight']: drop_edit(n)
apply_units([(89,U_ins),(170,U_ne)]); print('edited',edited())
open('/work/notes.md','a').write("\n## Session 9 (runs 245-251)\n- Tool rollout (orig) ~1.5-1.9x teacher-forced direct (M) for ins; NE rollout < TF direct. Effective rollout feedback rho~0.4 (not .85).\n- Ridge readout on WC (glucose rows, per (code,lastbin) centering, clip tilt targets): ins rho.4 lam30 oof frac .46 cost g1 .0018/glucose row, full .0107; NE rho.3 lam300 frac .31 cost full .0056/MAP row. Drug-bundle gating useless (88% rows).\n- codes: GF6 GL10 MA7 MC5 ins8 NE9. EDIT v1: unit89 (ins->GF/GL, vG tilt), unit170 (NE->MA/MC, vM), gate via embed.code min-var, linear GELU regime, centering in embed_transition rows; gains 1.5/1.5. fns fit_full, build_unit, apply_units.\n")
```