
import numpy as np, torch, json, math
torch.set_grad_enabled(False)
V = json.load(open('/model/vocab.json'))
CODE = {c['name']:c['id'] for c in V['codes']}
NUM = {(d['code'],d['attribute']):d for d in V['numeric']}
def fbin(code, attr, x):
    d=NUM[(code,attr)]; return 1+int(np.searchsorted(np.array(d['fine_bin_edges']), x, side='left'))
def zsc(code, attr, x):
    d=NUM[(code,attr)]; return float(np.clip((x-d['median'])/d['iqr'],-10,10))
EKIND = {'measurement':1,'drug_admin':2,'visit':3,'visit_detail':4}
def build_bundles(ev, opts):
    """ev: dataframe of one stay. returns list of (time_hours, atoms). atom=dict"""
    atoms=[]
    for r in ev.itertuples():
        e=r.event
        if e in ('glucose_finger','glucose_lab','map_arterial','map_cuff'):
            atoms.append((r.hour, dict(code=CODE[e],kind=0,ek=1,unit=opts['unit'][e],ru=0,val=(e,r.value),dose=None,rate=None)))
        elif e in ('insulin','norepinephrine'):
            atoms.append((r.hour, dict(code=CODE[e],kind=0,ek=2,unit=opts['unit'][e],ru=opts['ru'][e],val=None,dose=(e,r.dose),rate=(e,r.rate))))
            endat = dict(code=CODE[e],kind=2,ek=2,unit=opts['unit'][e] if opts['end_attrs'] else 0,ru=opts['ru'][e] if opts['end_attrs'] else 0,val=None,
                         dose=(e,r.dose) if opts['end_attrs'] else None, rate=(e,r.rate) if opts['end_attrs'] else None)
            atoms.append((r.end_hour, endat))
        else:
            ek = 3 if e=='hospital_admission' else 4
            atoms.append((r.hour, dict(code=CODE[e],kind=0,ek=ek,unit=0,ru=0,val=None,dose=None,rate=None)))
            atoms.append((r.end_hour, dict(code=CODE[e],kind=2,ek=ek,unit=0,ru=0,val=None,dose=None,rate=None)))
    atoms.sort(key=lambda a:a[0])
    bundles=[]; 
    for t,a in atoms:
        tm=round(t*60)
        if bundles and bundles[-1][0]==tm: bundles[-1][1].append(a)
        else: bundles.append((tm,[a]))
    return [(tm/60.0, al) for tm,al in bundles]
class M:
    def __init__(self, W): self.W={k:torch.tensor(np.asarray(v)) for k,v in W.items()}
    def lin(self, x, name, bias=True):
        y = x @ self.W[name+'.weight'].T
        if bias and (name+'.bias') in self.W: y = y + self.W[name+'.bias']
        return y
    def mlp(self, x, name):
        return self.lin(torch.nn.functional.gelu(self.lin(x, name+'.0')), name+'.2')
    def ln(self, x, name, eps=1e-5):
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), self.W[name+'.weight'], self.W[name+'.bias'], eps)
    def embed_atom(self, a, opts):
        W=self.W; v = W['embed.code.weight'][a['code']] + W['embed.kind.weight'][a['kind']] + W['embed.ekind.weight'][a['ek']]
        v = v + W['embed.categorical.unit.weight'][a['unit']] + W['embed.categorical.rate_unit.weight'][a['ru']]
        if opts.get('kq',True): v = v + W['embed.kq.weight'][0]
        num = torch.zeros(6)
        if a['val'] is not None:
            c,x=a['val']; v = v + W['embed.bins.value.weight'][fbin(c,'value',x)]; num[0]=zsc(c,'value',x); num[1]=1
        if a['dose'] is not None:
            c,x=a['dose']; v = v + W['embed.bins.dose.weight'][fbin(c,'dose',x)]; num[2]=zsc(c,'dose',x); num[3]=1
        if a['rate'] is not None:
            c,x=a['rate']; v = v + W['embed.bins.rate.weight'][fbin(c,'rate',x)]; num[4]=zsc(c,'rate',x); num[5]=1
        return v + W['embed.numeric.weight'] @ num + W['embed.numeric.bias']
    def encode_set(self, X, name):
        n = X.shape[0]
        s = self.mlp(X, name+'.mlp1').sum(0)/math.sqrt(n) + self.lin(torch.tensor([math.log(1+n)]), name+'.size')
        return self.ln(self.mlp(s, name+'.mlp2'), name+'.norm')
    def encode_bundle(self, atoms, opts):
        obs=[a for a in atoms if a['ek'] in (1,3,4)]; act=[a for a in atoms if a['ek']==2]
        e = self.encode_set(torch.stack([self.embed_atom(a,opts) for a in obs]), 'encode') if obs else torch.zeros(192)
        u = self.encode_set(torch.stack([self.embed_atom(a,opts) for a in act]), 'encode_action') if act else torch.zeros(192)
        return e,u
    def ctx(self, out, h, dt_s):
        decay = torch.exp(-dt_s*torch.exp(self.W['core.flow.a']))
        hf = h*decay
        g = self.mlp(torch.tensor([math.log(max(dt_s,1.0))/10.0]), 'gap')
        return out + self.ln(hf.mean(0),'core.out_norm') + g, hf
    def zpost(self, ctx, e, u):
        o = self.mlp(torch.cat([ctx,e,u]), 'z_post'); return o[:32], o[32:]
