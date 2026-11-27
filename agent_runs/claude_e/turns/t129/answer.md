Gating by "drug in the preceding bundle" helps the glucose side a lot:

| Glucose rows | Cost | Share of total cost |
|---|---|---|
| Finger, no insulin before | 0.068 per row | 25% |
| Lab, no insulin before | 0.015 per row | 3% |
| Finger, insulin before | 0.001 per row | 2% |

For finger rows after insulin the shift noise is also much smaller (mean squared shift 2.0, against 11.7 on rows with no insulin before).

The norepinephrine side is different. Its cost sits where the drug is running: cuff MAP 0.055 per row, arterial MAP 0.016. Cuff MAP readings carry 43% of the total cost, and the evaluation scores arterial MAP only.

So the redesign is:
- gate the insulin correction on "insulin in the preceding bundle";
- apply the norepinephrine correction to arterial MAP only;
- then calibrate a_ins and a_ne with a rollout.

Next I will check whether "previous bundle held insulin" and "previous bundle held norepinephrine" are linearly separable in the value head's 192-dim input. I use a closed-form ridge discriminant on the stored, z-averaged inputs and report AUC and class overlap. Clean separation with a wide margin would confirm a gate is buildable. Heavy overlap would rule it out.

```python
from sklearn.metrics import roc_auc_score
Rcat=pd.concat([datasets[i][1] for i in range(5)],ignore_index=True)
print('align stays/hours with RA_:', (Rcat['stay'].values==RA_['stay'].values).all(), (Rcat['hour'].values==RA_['hour'].values).all(), WA_.shape)
A_['stay']=Rcat['stay'].values
def lda_gate(mask,lab,lam=1e-2):
    Xg=WA_[mask].astype(np.float64); yg=lab[mask].astype(bool)
    mu1=Xg[yg].mean(0); mu0=Xg[~yg].mean(0); S=np.cov(Xg.T)+lam*np.eye(Xg.shape[1])
    w=np.linalg.solve(S,mu1-mu0); p=Xg@w
    c=(np.percentile(p[yg],1)+np.percentile(p[~yg],99))/2
    sd=p.std()
    print('  n',mask.sum(),'pos',yg.sum(),'AUC %.4f'%roc_auc_score(yg,p),
          'pos 0.1/1/5%%: %s'%np.round((np.percentile(p[yg],[0.1,1,5])-c)/sd,2),
          'neg 95/99/99.9%%: %s'%np.round((np.percentile(p[~yg],[95,99,99.9])-c)/sd,2))
    return w,c,sd
print('insulin gate on glucose rows (6,10):'); gi_=lda_gate(A_.code.isin([6,10]).values, A_.hi.values)
print('NE gate on arterial rows (7):'); gn_=lda_gate((A_.code==7).values, A_.hn.values)
print('NE gate on all MAP rows (7,5):'); gn2_=lda_gate(A_.code.isin([7,5]).values, A_.hn.values)
```