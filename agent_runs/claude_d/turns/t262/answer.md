The z-posterior reconstruction does not match. The mean error (0.23) is as large as the z values themselves (0.25), and it is nearly the same for every unit-id combination. So the problem is not the unknown ids but something more basic: my ctx, the meaning of the stored z, or an off-by-one in which state the posterior belongs to.

Before spending more runs on the encoder, I will localise the mismatch by comparing the stored z with five alternatives:
1. z_post computed with u_k set to zero, which shows whether u_k matters at all;
2. the prior mean z_prior(ctx);
3. z_post using the current row's state instead of the previous one;
4. z_post using the next bundle's state;
5. z_post with the input order of e_k and u_k swapped.

The variant with near-zero error reveals the right convention.

```python
def zpo(c, ek, uk): return MLPo(np.r_[c, ek, uk][None], 'z_post')[0][:32]
def zpr(c): return MLPo(c[None], 'z_prior')[0][:32]
uk0 = [uenc([avec(9, 0, rt, ds, 'norepinephrine', 4, 3)]) for (_,_,rt,ds) in rows_]
out_ = {'post_my_uk':[], 'post_uk0':[], 'prior':[], 'post_ctx_cur':[], 'post_next':[], 'swap_eu':[]}
for (j,c,rt,ds), uk in zip(rows_, uk0):
    zt = XALL[j, 960:992]
    out_['post_my_uk'].append(np.abs(zpo(c, np.zeros(192), uk)-zt).mean())
    out_['post_uk0'].append(np.abs(zpo(c, np.zeros(192), np.zeros(192))-zt).mean())
    out_['prior'].append(np.abs(zpr(c)-zt).mean())
    dtn = float((STALL.minute.values[j]-STALL.prev_min.values[j])*60)
    out_['post_ctx_cur'].append(np.abs(zpo(ctx_of(j, max(dtn,60.0)), np.zeros(192), uk)-zt).mean())
    if j+1 < len(XALL) and STALL.stay_id.values[j+1]==STALL.stay_id.values[j]: out_['post_next'].append(np.abs(zpo(c, np.zeros(192), uk)-XALL[j+1, 960:992]).mean())
    out_['swap_eu'].append(np.abs(zpo(c, uk, np.zeros(192))-zt).mean())
print({k: round(float(np.mean(v)),4) for k,v in out_.items()})
print('prev_row==j-1 frac', round(float(np.mean([int(STALL.prev_row.values[j])==j-1 for (j,_,_,_) in rows_])),2))
```