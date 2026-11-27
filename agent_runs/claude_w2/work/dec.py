
def lag_abs(DG,a,si,tau,L=12):
    X=np.zeros((len(si),L))
    for k in range(1,L+1):
        ix=tau-k+3; ok=ix>=0
        X[ok,k-1]=DG[a[ok],si[ok],ix[ok]]
    return X
def wc_hidden(D_, W, idx):
    base=D_['base'][idx]; codes=torch.as_tensor(D_['code'][idx],dtype=torch.long)
    lg,wc=head_value_logits(base,codes,D_['f8'][idx],D_['lb'][idx],torch.zeros(len(idx),dtype=torch.long),W,D_['zmu'][idx],1)
    h1=F.gelu(F.linear(wc,W['attr_heads.value_bin.0.weight'],W['attr_heads.value_bin.0.bias']))
    return wc,h1,lg
def lag_feats(R_, D_, arms, pfn, idx, L=12, drug='insulin'):
    DG=dose_stack(R_, arms, drug)
    hk=make_dev_oracle(R_, DG, np.zeros(L), t_hide=1e9, pf=pfn)
    si=D_['si'][idx]; a=D_['arm'][idx]; tau=np.round(D_['t'][idx]).astype(int)
    dev=hk.devs(D_['n'][idx], si, a, tau, np.zeros((1,1),int), L)
    d=lag_abs(DG, a, si, tau, L)
    return d, d-dev, dev
