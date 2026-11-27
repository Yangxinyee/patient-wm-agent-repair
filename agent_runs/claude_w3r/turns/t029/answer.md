Switching to the cross-fitted insulin innovation and richer controls changed the answer noticeably. Lag 1 went from −1.95 to −1.46 (unweighted) or −1.77 (IPW), and the tail turned positive after lag 9 (+0.2 to +0.37). The cumulative response is −8.9 at 8 h, falling back to −7.7 at 12 h. The two approaches disagree by more than their standard errors, so one innovation definition is contaminated.

The insulin policy is known to be an exact step function of the last reading plus i.i.d. noise, with bins aligned at 150/180/220/260/300. The band cell-mean deviation should therefore be the cleaner one, and the gradient-boosting residual can leak in the policy's predictable part. Next I will run a controlled comparison on one sample: hours with glucose measured at t and ≥ 155 (insulin always on), IPW for lags ≥ 2. I will compare the band deviation with few controls, the band deviation with full controls, and the boosting residual. If the band versions agree with each other, the band deviation is the valid design.

```python
IP['g5']=np.floor(IP.gl0/5)*5
IP['xb']=IP.r-IP.groupby('g5').r.transform('mean')
S = IP[(IP.gage==0)&(IP.gl0>=155)].copy()
def irf_generic(S, xcol, ctrl, maxs=12):
    red=[];se=[];drr=[]
    for s in range(0,maxs+1):
        d=S.copy(); tt=d.t.values+s; ok=tt<T; d=d[ok].copy(); tt=tt[ok]; d['one']=1.0
        d['ry']=ins_r[d.stay.values,tt]; bq,_=wols(d,'ry',[xcol]+ctrl,'one'); drr.append(bq[xcol])
        if s>0:
            d['y']=gf[d.stay.values,tt]; d['w']=np.where(ins_r[d.stay.values,tt-1]>0,1.0,1/0.30)
            b,sb=wols(d,'y',[xcol]+ctrl,'w'); red.append(b[xcol]); se.append(sb[xcol])
    red=np.array(red); drr=np.array(drr); h=np.zeros(maxs+1)
    for s in range(1,maxs+1): h[s]=(red[s-1]-sum(h[s-m]*drr[m] for m in range(1,s)))/drr[0]
    return red, np.array(se), h[1:]
for lab,xc,ct in [('band-x few ctrl','xb',['gl0','g02']),('band-x full ctrl','xb',['gl0','g02','g1','g2','r1','r2']),('hgb-x full','x',['gl0','g02','g1','g2','r1','r2'])]:
    red,se,h=irf_generic(S,xc,ct)
    print(lab,'\n red',np.round(red,2),'\n se ',np.round(se,2),'\n h  ',np.round(h,2),'\n cum',np.round(np.cumsum(h),1))
```