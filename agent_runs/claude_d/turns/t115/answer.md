Run 103 built the "last value of the same code" register: 93% of readings have an earlier value of their code.

Next is the first full head pass. It chains the prior mean of z, the view, the query LayerNorm for each of the 15 types, the code embedding plus registers, the value MLP and the transition table, then the softmax mean over the 16 bin centres. I'll report the mean absolute error against fwd_4's predicted means on glucose_finger for each type. One type with a much smaller error would identify the query type and tell me how far off the remaining choices are.

```python
gapf=lambda x: Lin(gelu(Lin(x[:,None],'gap.0')),'gap.2')
te=Wn('type_emb.weight'); ce=Wn('embed.code.weight'); bv=Wn('embed.bins.value.weight'); br=Wn('embed.bins.rate.weight')
TR=Wn('embed_transition.value_bin.weight')
def fbin_(code,v,side='right'):
    e=np.array(NUM[(code,'value')]['fine_bin_edges']); return 1+np.searchsorted(e,v,side=side)
def head(df,typ,gs=10.0,regv=1,side='right',renorm=True,zmode='mean'):
    i=df.prev.values; dt=df.dt_s.values
    ms=(HH[i]*np.exp(-dt[:,None,None]*fa[None,None,:])).mean(1); on=LNf(ms,'core.out_norm')
    ctx=OUT[i]+on+gapf(np.log(dt)/10); zp=Lin(gelu(Lin(ctx,'z_prior.0')),'z_prior.2'); zm=zp[:,:32]
    view=OUT[i]+on+gapf(np.log(dt)/gs)+Lin(zm,'z_head')
    q=LNf(view+te[typ],'query_norm')
    cid=df.event.map(code_id).values; z=np.zeros(len(df)); lb=np.zeros(len(df),int)
    for c in code_id:
        mk=(df.event.values==c)&(df.has_last.values>0); n=NUM[(c,'value')]
        z[mk]=(df.last_val.values[mk]-n['median'])/n['iqr']; lb[mk]=fbin_(c,df.last_val.values[mk],side)
    has=df.has_last.values; la=np.log1p(df.age_s.values)/10*has
    inp=np.zeros((len(df),8)); inp[:,0]=has; inp[:,1]=la; inp[:,2]=z
    if regv==2: inp[:,3]=has
    wc=q+ce[cid]+Lin(inp,'last_numeric')+bv[lb]+br[0]
    lg=Lin(gelu(Lin(wc,'attr_heads.value_bin.0')),'attr_heads.value_bin.2')+TR[cid*17+lb]
    p=np.exp(lg-lg.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
    cen=np.stack([vbc[e] for e in df.event.values]); pb=p[:,1:]
    if renorm: pb=pb/pb.sum(1,keepdims=True)
    return (pb*cen).sum(1), p
gfm=f4[f4.event=='glucose_finger']
print([round(np.mean(np.abs(head(gfm,t)[0]-gfm.predicted_mean.values)),2) for t in range(15)])
```