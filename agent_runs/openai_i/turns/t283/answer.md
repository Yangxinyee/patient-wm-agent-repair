The private innovations are effectively zero on untreated bundles, including admission, which supports the channel’s gating. The dose test stopped because the helper returns means rather than a two-item result. I’ll adapt to that return format and capture its logits separately. A stable dose response with low factual drift would justify a GPU test.

```python
from scipy.special import log_softmax
readout_saved252 = readout214
capture252 = {}
def readout_capture252(*args, **kwargs):
    result = readout_saved252(*args, **kwargs)
    capture252["result"] = result
    return result

def get_means252(result):
    if isinstance(result, (np.ndarray, torch.Tensor)):
        return array196(result)
    for x in result:
        if isinstance(x, (np.ndarray, torch.Tensor)) and tuple(x.shape) == (n182,4):
            return array196(x)
    raise ValueError("Unexpected mean return format")

def get_logits252(result):
    choices = result if isinstance(result,(tuple,list)) else [result]
    for x in choices:
        if isinstance(x,(np.ndarray,torch.Tensor)) and x.ndim >= 2 and x.shape[-1] == 17:
            return array196(x)
    raise ValueError("No fine-bin logits captured")

directstats252 = []
readout214 = readout_capture252
try:
    meanbase252 = get_means252(candidate_readout216(np.zeros((192,2),np.float32),(1.,1.)))
    logitsbase252 = get_logits252(capture252["result"])
    print("shapes",meanbase252.shape,logitsbase252.shape)
    lpbase252 = log_softmax(logitsbase252,axis=-1)
    pbase252 = np.exp(lpbase252)
    rates252 = np.asarray(R193)[np.asarray(ids182,dtype=int)]
    for gain in (0.1,0.3,1.0,3.0):
        start = time.time()
        vectors = vec251 * gain
        mf = get_means252(candidate_readout216(vectors,(1.,1.)))
        lf = get_logits252(capture252["result"])
        kl = float(np.mean(np.sum(pbase252*(lpbase252-log_softmax(lf,axis=-1)),axis=-1)))
        slopes = []
        for di,scales in enumerate(((2.,1.),(1.,2.))):
            me = get_means252(candidate_readout216(vectors,scales))
            active = rates252[:,di]>0
            slopes.append(np.sum(me[active]-mf[active],axis=0)/np.sum(rates252[active,di]))
        row = {"gain":gain,"KL_proxy":round(kl,7),
               "factual_mean_RMS":round(float(np.sqrt(np.mean((mf-meanbase252)**2))),5),
               "insulin_slopes":np.round(slopes[0],4).tolist(),
               "NE_slopes":np.round(slopes[1],4).tolist(),
               "seconds":round(time.time()-start,2)}
        directstats252.append(row)
        print(row)
finally:
    readout214 = readout_saved252

with open("/work/notes.md","a") as f:
    f.write("\n## Python 252 — repaired direct-token finite-dose checks\n"
            "- Untreated private innovations are effectively zero, including all retained admission bundles.\n"
            "- Run251 stopped on helper return-format handling; no weights changed.\n"
            "- Capturing logits without altering model computation; local diagnostics require later GPU certification.\n")
    for row in directstats252:
        f.write(str(row)+"\n")
```