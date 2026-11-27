Run 93 shows the surprise is not stored in the mean or scale themselves (R² about 0 from those two features). Normalizing the state destroys its linear readability, though (0.60 raw versus 0.01 after LayerNorm). The likeliest explanation is that the scale of the state varies a lot across stays. Dividing by it then multiplies the dose component by a varying factor, and the heads see the dose effect with a gain that depends on the stay. I'll check how widely the per-vector std ranges. I'll also check whether the raw state minus its mean, with scale kept, still decodes the surprise. That tells me whether an edit must act before normalization, in the core, rather than in the heads.

```python
print('sd pct',np.percentile(sd,[1,10,50,90,99]).round(2))
for off in [0,1]:
    ok=[i for i in src if i+off<len(X) and sid3[i+off]==sid3[i]]; rows=np.array(ok)+off
    t=T0[ok]; s=sid3[ok]; ye=np.array([RB[a][0][b]-RB[a][1][b] for a,b in zip(s,t)])
    Xbak=X; X=np.c_[Xm-mu[:,None], Xln/ np.median(1/sd), Xm]
    print('off',off,'raw-mu',r2cv(rows,ye,slice(0,192)),'raw',r2cv(rows,ye,slice(384,576)),'sd at rows pct',np.percentile(sd[rows],[10,50,90]).round(2))
    X=Xbak
```