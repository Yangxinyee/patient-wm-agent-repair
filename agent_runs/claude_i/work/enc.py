
EK = {c["name"]: {"measurement":1,"drug_admin":2,"visit":3,"visit_detail":4}.get(c.get("event_kind"),0) for c in VOCAB["codes"]}
def atom_matrix(atoms, cfg):
    W = {k: Tn(k) for k in ['embed.code.weight','embed.kind.weight','embed.ekind.weight','embed.kq.weight','embed.categorical.unit.weight',
         'embed.categorical.rate_unit.weight','embed.bins.value.weight','embed.bins.rate.weight','embed.bins.dose.weight','embed.numeric.weight','embed.numeric.bias']}
    V = np.zeros((len(atoms),192))
    for i,a in enumerate(atoms):
        c = a['code']; v = W['embed.code.weight'][CID[c]] + W['embed.kind.weight'][a['kind']] + W['embed.ekind.weight'][EK[c]] + W['embed.kq.weight'][0]
        num = np.zeros(6)
        if c in ('glucose_finger','glucose_lab','map_arterial','map_cuff'):
            v = v + W['embed.categorical.unit.weight'][cfg['unit'][c]]
            num[0] = zsc(c,'value',a['value']); num[1] = 1
            v = v + W['embed.bins.value.weight'][fbin(c,'value',np.array([a['value']]))[0]]
        elif c in ('insulin','norepinephrine'):
            v = v + W['embed.categorical.unit.weight'][cfg['unit'][c]] + W['embed.categorical.rate_unit.weight'][cfg['runit'][c]]
            if a['kind']==0 or cfg['end_attrs']:
                num[2] = zsc(c,'dose',a['dose']); num[3]=1; num[4] = zsc(c,'rate',a['rate']); num[5]=1
                v = v + W['embed.bins.dose.weight'][fbin(c,'dose',np.array([a['dose']]))[0]] + W['embed.bins.rate.weight'][fbin(c,'rate',np.array([a['rate']]))[0]]
        v = v + W['embed.numeric.weight'] @ num + W['embed.numeric.bias']
        V[i] = v
    return V
def set_encode(V, pre):
    n = V.shape[0]
    s = mlp2(V, pre + '.mlp1').sum(0) / np.sqrt(n) + Tn(pre + '.size.weight')[:,0]*np.log1p(n) + Tn(pre + '.size.bias')
    return lnorm(mlp2(s[None,:], pre + '.mlp2')[0], Tn(pre + '.norm.weight'), Tn(pre + '.norm.bias'))
def bundle_eu(atoms, cfg):
    ms = [a for a in atoms if a['code'] not in ('insulin','norepinephrine')]
    ds = [a for a in atoms if a['code'] in ('insulin','norepinephrine')]
    e = set_encode(atom_matrix(ms,cfg), 'encode') if ms else np.zeros(192)
    u = set_encode(atom_matrix(ds,cfg), 'encode_action') if ds else np.zeros(192)
    return e, u
