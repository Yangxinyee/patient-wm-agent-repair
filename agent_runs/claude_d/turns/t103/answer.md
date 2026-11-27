With correct alignment, the surprise of a dose is readable from the four state slots right after its bundle (R² 0.64). It falls to 0.43 one bundle later, 0.12 two bundles later and about 0 after that. It is not readable from `out` (R² 0.01). The value head reads `out` plus a normalized mean of the slots, so the surprise may sit in slot-specific directions that the head never sees. It also fades within about two hours, while the true effect lasts about 12.

Next I'll check whether the slot mean, which the head does see, carries it, and which slot holds it. If the slot mean has R² near 0 while individual slots are high, the head is blind to it. A fix then needs a readout of the slots or a core edit that carries it forward.

```python
for off in [0,1]:
    ok=[i for i in src if i+off<len(X) and sid3[i+off]==sid3[i]]; rows=np.array(ok)+off
    t=T0[ok]; s=sid3[ok]; ye=np.array([RB[a][0][b]-RB[a][1][b] for a,b in zip(s,t)])
    Hs=X[:,192:960].reshape(len(X),4,192); Xm=Hs.mean(1)
    Xbak=X; X=np.c_[Xm, Hs[:,0], Hs[:,1], Hs[:,2], Hs[:,3]]
    print('off',off,'mean_slots',r2cv(rows,ye,slice(0,192)),'slots',[r2cv(rows,ye,slice(192*(k+1),192*(k+2))) for k in range(4)])
    X=Xbak
```