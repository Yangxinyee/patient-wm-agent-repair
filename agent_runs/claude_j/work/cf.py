
def drug_hist(s,drug):
    dd={}
    for (mnt,evs) in BUN[s]:
        for (ev,kd,row) in evs:
            if ev==drug and kd=='start': dd[int(round((mnt-5)/60))]=float(row.rate)
    return dd
GLU=('glucose_finger','glucose_lab'); MAPC=('map_arterial','map_cuff')
def struct_delta(s,minute,codeset,f,start_h=12):
    glu=codeset==GLU; drug='insulin' if glu else 'norepinephrine'; a=0.6 if glu else 0.39; gain=-2.0 if glu else 3.3
    dd=drug_hist(s,drug); tau=None
    for (mnt,evs) in BUN[s]:
        if mnt>=minute: break
        if any(ev in codeset and kd=='world' for (ev,kd,row) in evs): tau=mnt
    t=int(round(minute/60)); th=int(round(tau/60)) if tau is not None else -99
    dx=0.0; tot=0.0
    for m in range(start_h+1,t+1):
        dx=a*dx+(f-1)*dd.get(m-1,0.0)*(m-1>=start_h)
        if m>th: tot=tot*a+dx
        else: tot=0.0
    # tot accumulates sum_{m>th} a^{t-m} dx_m
    return gain*tot
def heads_on_states(O,H,s,rt,eps):
    # rt: reading-table rows of stay s (b>=1); eps: (n,4,32) fixed draws
    b=torch.tensor(rt['b'].values); dt=torch.tensor(rt['dt'].values,dtype=torch.float64)
    out=O[b-1]; h=H[b-1]
    hf=h*torch.exp(-dt[:,None,None]*torch.exp(HT('core.flow.a')))
    ctx=out+Hln(hf.mean(1),'core.out_norm')+Hlin(Fn.gelu(Hlin(torch.log(dt)[:,None]/10,'gap.0')),'gap.2')
    zp=Hlin(Fn.gelu(Hlin(ctx,'z_prior.0')),'z_prior.2')
    z=zp[:,None,:32]+torch.exp(zp[:,None,32:])*eps
    q=Hln(ctx[:,None,:]+Hlin(z,'z_head')+HT('type_emb.weight')[1],'query_norm')
    f,lb,cid=last_feats(rt); wc=with_code(q,f,lb,cid); P=value_probs(wc,lb,cid)
    cen=torch.tensor(np.stack([CEN[c] for c in rt['event']]))
    return q,wc,P,(P[:,1:]*cen).sum(1)/P[:,1:].sum(1)
