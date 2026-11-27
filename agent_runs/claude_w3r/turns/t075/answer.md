The best type disagrees between events: row 10 for glucose (2.85 mg/dL median error) and row 1 or 2 for MAP (about 2.7 mmHg). The type should be the same for all measurements, so the head is still structurally off. With type 1 the MAP error is moderate but the glucose error is 6.7 mg/dL. That points to something specific to the register inputs, either the positions of the 8 last_numeric inputs or the unit of the age term.

Next I fix type 1, the likely "measurement WORLD" type, and search the register options jointly over glucose and MAP:
- where has_last, log age, z_last and an optional has_value flag go among the 8 inputs;
- age in seconds, minutes or hours;
- last_status 0 or 1.

A combination that gets both events down to about 1 unit or less would validate the head. If none does, the problem is elsewhere, for example in the view or z.

```python
def prep(HI_, code_name, ns=32):
    P=[]
    g_=torch.Generator().manual_seed(0)
    for (i0,dt,lv,age) in HI_:
        out_=torch.tensor(S_['x'][i0,:192]); h_=torch.tensor(S_['x'][i0,192:960]).reshape(4,192)
        hf=h_*torch.exp(-dt*torch.exp(fa)); ctx=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([math.log(dt)/10],dtype=torch.float32),'gap','gelu')
        pr=MLP(ctx,'z_prior','gelu'); zs=pr[:32]+torch.exp(pr[32:])*torch.randn(ns,32,generator=g_)
        P.append((ctx+L_(zs,'z_head'), lv, age))
    return P
PG=prep(HI[:60],'glucose_finger'); PM=prep(HM,'map_arterial')
def pm_from(P, code_name, typ, layout, lstat, age_div):
    c_=CODE[code_name]; cen=torch.tensor(VOCAB['value_bin_centres'][code_name],dtype=torch.float32); res_=[]
    for view,lv,age in P:
        q=LN(view+Wo['type_emb.weight'][typ],'query_norm'); num=torch.zeros(8); lb=0; extra=0
        if lv is not None:
            lb=fbin(code_name,'value',lv); vals=[1.0, math.log(1+age/age_div)/10, zsc(code_name,'value',lv), 1.0]
            for pos,val in zip(layout,vals):
                if pos is not None: num[pos]=val
            extra=Wo['last_status.weight'][lstat]
        wc=q+Wo['embed.code.weight'][c_]+L_(num,'last_numeric')+Wo['embed.bins.value.weight'][lb]+extra
        lg=MLP(wc,'attr_heads.value_bin','gelu')+Wo['embed_transition.value_bin.weight'][c_*17+lb]
        res_.append(float((torch.softmax(lg,-1).mean(0)[1:]*cen).sum()))
    return np.array(res_)
cands=[]
for lay in [(0,1,2,None),(0,1,2,3),(0,1,2,4),(0,1,3,None),(0,2,1,None),(0,1,4,None),(0,1,2,5),(0,1,2,6),(0,1,2,7),(1,0,2,None)]:
    for lstat in [0,1]:
        for ad in [1.0,60.0,3600.0]:
            eg=np.median(np.abs(pm_from(PG,'glucose_finger',1,lay,lstat,ad)-obs_pm[:60])); em=np.median(np.abs(pm_from(PM,'map_arterial',1,lay,lstat,ad)-fm.predicted_mean.values))
            cands.append((round(eg+em,2),round(eg,2),round(em,2),lay,lstat,ad))
cands.sort(key=lambda t:t[0]); [print(c) for c in cands[:8]]
```