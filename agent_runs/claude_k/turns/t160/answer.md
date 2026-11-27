No z_post unit is dead on both kinds of bundle. The units that are exactly zero on drug bundles do produce small, non-zero activations on reading bundles, and my "exactly dead" test (below 10⁻⁶) was too strict. Nothing was saved, because the run stopped before any edit.

So I'll rank units by their actual importance on both kinds of bundle: the standard deviation of the activation times the size of the unit's effect on the z token. I'll take the two with the smallest worst case. When repurposing them I'll fold their mean output into the z_post output bias, so the only thing lost is their tiny variation.

If two units have importance near 0.001 or less on both kinds of bundle (median is 0.0004), I'll save the writer edit and run the same offline check. It should show injections of exactly 0.3 × insulin rate into z₁₈ and 2 × norepinephrine rate into z₂₈, with other dimensions changed only by about the removed units' variation.

```python
Ar_std=Ar.std(0); imp_r=Ar_std*eff; imp_d=imp
score=np.maximum(imp_d,imp_r); o_=np.argsort(score)
print('best units (score, imp_drug, imp_read, mean_act_read, max|act|):',[(int(j),round(score[j],5),round(imp_d[j],5),round(imp_r[j],5),round(Ar[:,j].mean(),4),round(max(np.abs(Ar[:,j]).max(),np.abs(Az[:,j]).max()),3)) for j in o_[:6]])
j_ins,j_ne=int(o_[0]),int(o_[1]); Bz=5.0; lam_i,lam_n=0.3,2.0; zd_i,zd_n=18,28
nd,nr=len(Az),len(Ar)
W0n,b0n,W2n,b2n=W0z.copy(),b0z.copy(),W2z.copy(),b2z.copy()
for j,p,lam,zd in [(j_ins,p_ins,lam_i,zd_i),(j_ne,p_ne,lam_n,zd_n)]:
    mact=(Az[:,j].sum()+Ar[:,j].sum())/(nd+nr)
    b2n+=W2z[:,j]*mact
    W0n[j]=0; W0n[j,384:576]=p; b0n[j]=Bz; W2n[:,j]=0; W2n[zd,j]=lam; b2n[zd]-=lam*gelu(Bz)
zo=mlpf(Zin_d,'z_post')[:,:32]; zn=(gelu(Zin_d@W0n.T+b0n)@W2n.T+b2n)[:,:32]; dz=zn-zo
print('drug: dz18 slope on ins rate %.4f corr %.4f | dz28 slope on ne %.4f corr %.4f | other dims max|dz| %.4f'%(np.polyfit(YA[s0,0],dz[:,18],1)[0],np.corrcoef(dz[:,18],YA[s0,0])[0,1],np.polyfit(YA[s0,1],dz[:,28],1)[0],np.corrcoef(dz[:,28],YA[s0,1])[0,1],np.abs(np.delete(dz,[18,28],1)).max()))
zr_n=(gelu(Zr@W0n.T+b0n)@W2n.T+b2n)[:,:32]; print('reading bundles max |dz|', np.abs(zr_n-mlpf(Zr,'z_post')[:,:32]).max().round(4))
for n_,a_ in [('z_post.0.weight',W0n),('z_post.0.bias',b0n),('z_post.2.weight',W2n),('z_post.2.bias',b2n)]: save_edit(n_, torch.tensor(a_,dtype=torch.float32))
print('edited', edited(), 'units', j_ins, j_ne)
```