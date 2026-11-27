`rd` holds the code name in `event` (along with stay, hour, value, last value and the previous-row pointer), and `with_code` was captured for all 14,296 readings, with factual NLL 1.6304 as before.

Now the free-gate test. I will fit a ridge least-squares discriminant on `with_code`: glucose_finger = +1, the other value codes = −1. The fit uses half the stays and the margin check the other half, across all 4 z samples. I will do the same for map_arterial, which the NE readout will need.

What decides it is whether the score ranges overlap. Clean separation with a gap of at least about 0.5 in target units means the gate can be built with a large multiplier and no extra killed dim. Any overlap rules it out and sends me to the encoder-cut gate dim.

```python
ev=rd['event'].values; st=rd['stay_id'].values
us=np.unique(st); tr_st=set(us[::2])
mtr=np.array([s in tr_st for s in st]); mte=~mtr
X=WC.numpy().astype(np.float64)
def disc(code):
    y=np.where(ev==code,1.0,-1.0)
    Xa=X[mtr].reshape(-1,192); ya=np.repeat(y[mtr],4)
    Xa1=np.c_[Xa,np.ones(len(Xa))]
    w=np.linalg.solve(Xa1.T@Xa1+1e-3*np.eye(193), Xa1.T@ya)
    s=(X[mte]@w[:192]+w[192])
    pos=s[ev[mte]==code].ravel(); neg=s[ev[mte]!=code].ravel()
    print(code, 'pos min/1%', round(pos.min(),3), round(np.percentile(pos,1),3), '| neg max/99%', round(neg.max(),3), round(np.percentile(neg,99),3), '| n', len(pos)//4, len(neg)//4)
    return w
print({c:int((ev==c).sum()) for c in np.unique(ev)})
w_gf=disc('glucose_finger'); w_ma=disc('map_arterial')
```