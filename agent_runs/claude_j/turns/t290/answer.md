Projecting the readout only off the last_numeric columns that are unused for measurements works best. Insulin keeps fraction 0.46 at out-of-fold cost 0.0017 per glucose row at gain 1, and 0.0073 at full gain. Drug B keeps fraction 0.31 at cost 0.0009 per MAP row. The raw readout spans only about 7 within-group sds, and its code-embedding dot products are small, so a gate of about ±10 separates cleanly.

I now build edit v2. Each drug gets a main unit (κ·w + gate) and a cancelling unit (gate only) with the opposite output column, so the gate noise drops out:
- insulin: unit 89 with 149 cancelling, on glucose codes;
- drug B: unit 170 with 118 cancelling, on MAP codes.

Factual group means go into embed_transition, and the gain is 1.5 for both drugs. Needed margins: every target-code row must activate at 5 or more, and every other row must stay at −4 or less (including estimates for unmeasured codes). I also look for an existing truth table for later checks.

```python
BVv,BRr=BT[2],BT[1]
def unit_pair(w,pos,fa_,v_,g,G=10.0,ut=-2.0):
    sr=np.std(Xw@w); kap=1.0/sr; meas=[5,6,7,10]
    t=np.array([1.0 if c in pos else (-1.0 if c in meas else ut) for c in range(13)]+[0.0]*(Cm.shape[0]-13))[nz]
    wg=Swi@Cz.T@np.linalg.pinv(Cz@Swi@Cz.T)@t; um=kap*w+G*wg; uc=G*wg; ip=np.isin(cidM,pos)
    sm=Xm@um; sc=Xm@uc; B=5.0-sm[ip].min(); Bc=5.0-sc[ip].min(); am=sm+B; ac=sc+Bc
    xr=Xm-EC[cidM]; l6=abs(kap*w@LNW[:,6])*10
    est=max(max(EC[c]@um+(xr@um).max()+B+l6, EC[c]@uc+(xr@uc).max()+Bc) for c in range(13) if c not in meas)
    print('kap',round(kap,1),'pos am',round(am[ip].min(),1),round(am[ip].max(),1),'ac',round(ac[ip].min(),1),round(ac[ip].max(),1),
          '| neg max am',round(am[~ip].max(),1),'ac',round(ac[~ip].max(),1),'| unmeas est max',round(est,1))
    col=(g/kap)*v_; d=WC[fa_.ri.values]@(um-uc)+B-Bc; cc=fa_.event.map(cid_of).values; lbv=fa_.lb.values; mu=np.zeros((13,17))
    for c in pos:
        mk=cc==c; bb=kap*(BVv+BRr)@w; base=d[mk].mean()-bb[lbv[mk]].mean()
        for lb in range(17):
            m2=mk&(lbv==lb); mu[c,lb]=d[m2].mean() if m2.sum()>=5 else base+bb[lb]
    return dict(um=um,B=B,uc=uc,Bc=Bc,col=col,mu=mu,pos=pos,kap=kap)
def apply_pairs(pairs):
    for n in ['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','embed_transition.value_bin.weight']: drop_edit(n)
    H0=L_('attr_heads.value_bin.0.weight'); b0=L_('attr_heads.value_bin.0.bias'); H2=L_('attr_heads.value_bin.2.weight'); ET=L_('embed_transition.value_bin.weight')
    for jm,jc,U in pairs:
        H0[jm]=U['um']; b0[jm]=U['B']; H2[:,jm]=U['col']; H0[jc]=U['uc']; b0[jc]=U['Bc']; H2[:,jc]=-U['col']
        for c in U['pos']:
            for lb in range(17): ET[c*17+lb]-=U['col']*U['mu'][c,lb]
    for n,a in [('attr_heads.value_bin.0.weight',H0),('attr_heads.value_bin.0.bias',b0),('attr_heads.value_bin.2.weight',H2),('embed_transition.value_bin.weight',ET)]:
        save_edit(n,torch.tensor(a,dtype=torch.float32))
w_ins,fr_i,_=fitw2(fac,cf,vG,0.4,30,0.0,Pd,1e-2); w_ne,fr_n,_=fitw2(faN,cN,vM,0.3,300,0.0,Pd,1e-2)
P_ins=unit_pair(w_ins,[6,10],fac,vG,1.5); P_ne=unit_pair(w_ne,[7,5],faN,vM,1.5)
apply_pairs([(89,149,P_ins),(170,118,P_ne)]); print('edited',edited())
print('R in kernel:',type(globals().get('R')).__name__, getattr(globals().get('R'),'shape',None))
open('/work/notes.md','a').write("- runs 252-257: readout must NOT be projected off LN cols 0,3,6 (cost x50); project only drug LN cols [1,2,4,5,7] (Pd). fitw2(fa,c,v,rho,lam,0,Pd,1e-2): ins frac .46 cost g1 .0017 full .0073; NE frac .31 g1 .0009 full .0056.\n- near-dead head units: 89,170,118,158,149. EDIT v2 (run 258): pairs main+cancel (89/149 ins, 170/118 NE), gate G=10 (min-var, orth to bins & LN), kap=1/sd, gains 1.5/1.5. fns unit_pair, apply_pairs.\n")
```