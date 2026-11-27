Run 285 told me what `mkW3` reads: it builds the edit from globals, including `GATE`, `DEC2` (dose decoder), `PH` (expected-dose model), `VD_` (injected directions) and `Pk` (copy-of-u basis). So I can swap in better injection directions through `VD_` without its source.

The gain sweep crashed in `bmlp` with "float is not callable". My own code caused this: I used `G_` as the loop variable, but `G_` was a global function the model code calls, apparently the activation inside `bmlp`. Earlier inventory runs also reused other global names (`fn`, `sig`, `s`, `cands`, ...).

Next I will:
- find where `G_` (and any other name I reused) is defined in the /work source files, and which kernel functions use those names;
- restore `G_` from its file definition, or by testing gelu variants;
- check the restore is exact against two known results: `tfnll(W0z)` = 1.5313237 and `Wt2`'s insulin ×2 readout = −3.697;
- only if both match, re-run the gain sweep at 3, 6 and 10 under non-clashing names.

If the restore is wrong, the check values won't match, and I'll stop before trusting any sweep numbers.

```python
import re
src_all={fn_: open('/work/'+fn_).read() for fn_ in os.listdir('/work') if fn_.endswith('.py')}
for fn_,s_ in src_all.items():
    for i,l in enumerate(s_.split('\n')):
        if re.search(r'(^|[^\w.])G_\b', l): print(fn_, i, l.strip()[:140])
over=['G_','fn','sig','s','k','v','r','t','stays','cands','g','lines','zo','bj','p','e','nm','args_','Wv','src','doc','c_','f_','r0','r1','r2','n_','SW','arrs','funcs','oth','diffs','Wt2','i','l','m']
def allnames(co):
    out=set(co.co_names)
    for cst in co.co_consts:
        if hasattr(cst,'co_names'): out|=allnames(cst)
    return out
refs={nm_:sorted(allnames(o_.__code__)&set(over)) for nm_,o_ in list(globals().items()) if isinstance(o_,types.FunctionType) and allnames(o_.__code__)&set(over)}
print('refs:', refs)
restored=False
for fn_,s_ in src_all.items():
    for l in s_.split('\n'):
        if re.match(r'^G_\s*=', l): exec(l, globals()); print('restored from', fn_, l); restored=True
cands_G=[('file',None)] if restored else []
cands_G+= [('gelu_t',gelu_t),('F.gelu',torch.nn.functional.gelu),('gelu',gelu)]
st40=list(BUN.keys())[:40]
for nmG,fG in cands_G:
    if fG is not None: G_=fG
    try:
        v_=tfnll(W0z,st40)[0]; print(nmG, 'tfnll W0z %.10f'%v_, 'match' if abs(v_-1.5313237216015931)<1e-7 else 'NO')
        if abs(v_-1.5313237216015931)<1e-7: break
    except Exception as ex: print(nmG,'fail',repr(ex)[:150])
r_=evalW(Wt2,st40); print('Wt2 check', r_[1][0], '(want -3.697)')
if abs(r_[1][0]+3.697)<0.002:
    SW={}
    for gg_ in [3.0,6.0,10.0]:
        Wgg=mkW3(gg_,gg_,W0z); rr_=evalW(Wgg,st40); nn_=tfnll(Wgg,st40); SW[gg_]=(rr_,nn_)
        print('gain',gg_, {kk:vv[0] for kk,vv in rr_.items()}, 'tfnll %.5f'%nn_[0])
```