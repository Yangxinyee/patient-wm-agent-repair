Adding history features barely changes the fit of the titration rule: residual variance is 0.01848 with current MAP and previous rate, and 0.01847 with everything added. So to first order the rule uses only current MAP and the previous rate.

Yet the minimal and full outcome models gave readouts of −1.05 versus −0.40. With a truly random innovation that cannot happen. The reason is scale: the innovation is tiny (sd 0.136) compared with how predictable future MAP is from history. A correlation of only about 0.02 between the innovation and history, far too small to show in the residual variance, then shifts the estimate by about 0.4–0.6.

One mechanism produces exactly this pattern: clinicians titrate on the patient's true MAP (a monitor they see), not on the charted noisy reading. The "innovation" would then carry information about the true MAP, biasing the estimate negative. More history pins down the true MAP better, so the bias shrinks (minimal −1.05 → full −0.40). It would also be larger when the current reading is the noisier cuff (−0.60 versus −0.31 for arterial), which is what I see. If this is right, the remaining −0.4 may be bias as well, and norepinephrine's true effect may be about 0.

I can test this directly from the titration rule. Two predictions would confirm it:
- In rows with both arterial and cuff readings, the cuff value still predicts the rate after controlling for arterial.
- The slope on cuff-only readings is weaker than on arterial-only readings, because noise dilutes it.

A rule based on the charted reading predicts neither. I will look at the shape of the rule by MAP and previous-rate bins, compare slopes across reading types, and check for norepinephrine given in hours with no MAP reading.

```python
t0=time.time()
r1v=NE0[si,ti-1]; rv=NE[si,ti]; dr=rv-r1v; ga=GA[si,ti]; gc=GC[si,ti]
both=~np.isnan(ga)&~np.isnan(gc); aonly=~np.isnan(ga)&np.isnan(gc); conly=np.isnan(ga)&~np.isnan(gc); none_=np.isnan(ga)&np.isnan(gc)
print('cont rows: both %d, art only %d, cuff only %d, none %d'%(both.sum(),aonly.sum(),conly.sum(),none_.sum()))
# shape of policy: mean dr by MAP bins (arterial-only rows) and r1 bins
mb=np.arange(50,96,3)
for nm,m,x in [('art-only',aonly,ga),('cuff-only',conly,gc)]:
    row=[]
    for lo in mb:
        mm=m&(x>=lo)&(x<lo+3); row.append('%d:%.3f'%(lo,dr[mm].mean()) if mm.sum()>200 else '%d:-'%lo)
    print(nm,' '.join(row))
for lo,hi in [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.2),(1.2,5)]:
    mm=aonly&(r1v>=lo)&(r1v<hi); print('  r1 in [%.1f,%.1f) n %d mean dr %.3f, sd dr %.3f; at MAP 60-66: %.3f, 70-76: %.3f'%(lo,hi,mm.sum(),dr[mm].mean(),dr[mm].std(),
          dr[mm&(ga>=60)&(ga<66)].mean(), dr[mm&(ga>=70)&(ga<76)].mean()))
# linear policy in a central region: dr ~ MAP + r1 (+ both readings)
from numpy.linalg import lstsq
def ols(cols,y):
    Xm=np.column_stack([np.ones(len(y))]+cols); b,_,_,_=lstsq(Xm,y,rcond=None); res=y-Xm@b
    cov=np.linalg.inv(Xm.T@Xm)*res.var(); return b,np.sqrt(np.diag(cov))
c=both&(ga>55)&(ga<85)&(gc>50)&(gc<90)
b,se=ols([ga[c],gc[c],r1v[c]],dr[c]); print('BOTH rows: coef art %.4f (%.4f) cuff %.4f (%.4f) r1 %.4f'%(b[1],se[1],b[2],se[2],b[3]))
c=aonly&(ga>55)&(ga<85); b,se=ols([ga[c],r1v[c]],dr[c]); print('ART-only: coef map %.4f (%.4f)'%(b[1],se[1]))
c=conly&(gc>55)&(gc<85); b,se=ols([gc[c],r1v[c]],dr[c]); print('CUFF-only: coef map %.4f (%.4f)'%(b[1],se[1]))
# HGB check: within BOTH rows, does adding cuff to (art, r1) reduce residual variance?
def rv_fit(cols,y,mask):
    X_=np.column_stack(cols)[mask]; yy=y[mask]; f_=((stay_ids[si[mask]]//7)%2); pred=np.zeros(len(yy))
    for f in (0,1):
        tr=f_!=f; te=f_==f; pred[te]=HGB(max_iter=300,min_samples_leaf=50).fit(X_[tr],yy[tr]).predict(X_[te])
    return np.var(yy-pred)
print('BOTH rows resid var: art+r1 %.5f ; art+cuff+r1 %.5f ; cuff+r1 %.5f'%(rv_fit([ga,r1v],rv,both),rv_fit([ga,gc,r1v],rv,both),rv_fit([gc,r1v],rv,both)))
print('time',round(time.time()-t0,1))
```