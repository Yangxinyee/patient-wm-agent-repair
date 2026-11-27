The non-zero last_status row and the extra columns are ruled out: adding them makes the fit worse. So the pipeline is right for readings with an empty MAP register, and the errors come from how the register is encoded when it is filled, plus something specific to the first glucose reading.

My earlier tests used only one column layout. Next I run a proper coordinate search over the 8 last_numeric columns. Each column is assigned one of: zero, has, log(1+age)/10 in hours, minutes or seconds, or the register z. The objective is the mean absolute error of predicted means against the tool. I start from the layout implied by ARCHITECTURE.md: [has, age, z, z_rate(=0), …].

If one layout brings the error to the noise floor (about 1.0), the register is solved. If the best stays near 2.3, something other than the column layout is wrong, such as the bin or z of the register value.

```python
q_fix=LN(ctx+F.linear(mu,W['z_head.weight'],W['z_head.bias'])+W['type_emb.weight'][1],W,'query_norm')
base_wc=q_fix+W['embed.code.weight'][codes_]+W['embed.bins.value.weight'][lb_]
trans=W['embed_transition.value_bin.weight'][codes_*17+lb_]
hasv=fa.has.values.astype(np.float32); ag=fa.age_s.values
FE={'zero':np.zeros(len(fa),np.float32),'has':hasv,'la_h':(np.log1p(ag/3600)/10*hasv).astype(np.float32),'la_m':(np.log1p(ag/60)/10*hasv).astype(np.float32),
    'la_s':(np.log1p(ag)/10*hasv).astype(np.float32),'zl':(fa.zl.values*hasv).astype(np.float32)}
def score(layout):
    f=np.stack([FE[c] for c in layout],1)
    wc=base_wc+F.linear(torch.as_tensor(f),W['last_numeric.weight'],W['last_numeric.bias'])
    pm=(torch.softmax(mlp(wc,W,'attr_heads.value_bin')+trans,-1)[:,1:].numpy()*cm).sum(1)
    return float(np.abs(pm-fa.predicted_mean.values).mean())
best_l=['has','la_h','zl','zero','zero','zero','zero','zero']; best_v=score(best_l); print('start',round(best_v,3))
for it in range(3):
    for c in range(8):
        for opt in FE:
            t_=list(best_l); t_[c]=opt; v_=score(t_)
            if v_<best_v-1e-4: best_v,best_l=v_,t_
    print(it,best_l,round(best_v,3))
```