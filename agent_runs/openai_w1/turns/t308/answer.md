The measured slot-only feedback is much smaller than the raw linear feedback, but it could still destabilize a strong edit. It also omits the posterior’s response to changed state. I’ll include that path and find injection directions that retain reading sensitivity with less feedback. This remains a local diagnostic; factual validation will decide whether the directions are safe.

```python
import time
t270 = time.time()
P270 = np.asarray(fits266[1.0], dtype=np.float64)
backupW270 = {n: rx_W[n] for n in edited()}
backupTokens270 = rx_tokens126
backupH270 = h269
backupE270 = E269

def posterior_token270(inp):
    q = rx_mlp(inp, "z_post", "gelu")
    return _rxlinear257(q[:, :32], "z_in")

try:
    inp270 = torch.as_tensor(np.asarray(ds_in)[rx_pick], dtype=torch.float32).clone()
    inp270[:,192:384] = rx_tokens126[:,4]
    inp270[:,384:576] = rx_tokens126[:,5]
    muold270 = rx_mlp(inp270, "z_post", "gelu")[:, :32]
    print("original posterior reconstruction RMS",
          float(torch.sqrt(torch.mean((muold270-rx_z)**2))))
    for n in backupW270:
        rx_W[n] = torch.as_tensor(load(n), dtype=torch.float32)

    raw270 = backupTokens270.clone()
    zin270 = posterior_token270(inp270)
    raw270[:,6] = zin270
    rx_tokens126 = raw270
    E270 = prelude269(raw270)
    print("current E versus cached fitted E RMS",
          float(np.sqrt(np.mean((E270.numpy()-np.asarray(factualE258)[rx_pick])**2))))
    xx270 = rx_hflow + E270
    uu270 = xx270
    for p in ("core.blocks.0.0", "core.blocks.0.1"):
        uu270 = rx_layer2(uu270,p,"gelu",memory=rx_mem,age=rx_ages,
                         valid=rx_memvalid,mode="before_k",age_act="gelu")
    h269 = a269*rx_hflow + d269*_rxlinear257(E270,"core.b_in") + uu270-xx270
    normh270 = rx_norm(rx_hflow.mean(1),"core.out_norm")
    eps270 = 0.10
    K270 = np.empty((len(rx_pick)*4*2,192),dtype=np.float64)
    for j in range(192):
        v = np.zeros(192,dtype=np.float32)
        v[j] = eps270
        yp,op = coda269(v)
        ym,om = coda269(-v)
        dh = (yp-ym)/(2*eps270)*flow269
        dc = ((op-om)/(2*eps270)
              + (rx_norm((rx_hflow+eps270*dh).mean(1),"core.out_norm")
                 -rx_norm((rx_hflow-eps270*dh).mean(1),"core.out_norm"))/(2*eps270))
        ip,im = inp270.clone(),inp270.clone()
        ip[:,:192] += eps270*dc
        im[:,:192] -= eps270*dc
        tp,tm = raw270.clone(),raw270.clone()
        tp[:,:4] += eps270*dh
        tm[:,:4] -= eps270*dh
        tp[:,6] += posterior_token270(ip)-zin270
        tm[:,6] += posterior_token270(im)-zin270
        de = (prelude269(tp)-prelude269(tm))/(2*eps270)
        K270[:,j] = (de.numpy()@P270).reshape(-1)

    covK270 = K270.T@K270/len(K270)
    eig270,vec270 = np.linalg.eigh(covK270)
    fitsB270 = {}
    print("feedback metric eigenvalue quantiles",
          np.round(np.quantile(eig270,[0,.5,.9,1]),5))
    for penalty in (0.,1.,10.,100.,1000.,10000.):
        invmetric = (vec270/(1+penalty*np.maximum(eig270,0)))@vec270.T
        A = invmetric@J257.T
        B = A@np.linalg.solve(J257@A + 1e-8*np.eye(4),T257)
        fitsB270[penalty] = B
        fb = (K270@B).reshape(len(rx_pick),4,2,2)
        print("penalty",penalty,"B RMS",np.round(np.sqrt(np.mean(B**2,axis=0)),3),
              "feedback RMS",np.round(np.sqrt(np.mean(fb**2,axis=(0,1))),3),
              "feedback p95",np.round(np.quantile(np.abs(fb),.95,axis=(0,1)),3))
        if penalty in (0.,100.,1000.):
            z = direct_readout257(np.zeros(192,dtype=np.float32))
            print("  finite one-unit glucose/MAP response",
                  np.round(np.stack([direct_readout257(B[:,j])-z for j in range(2)],axis=1),3))
finally:
    rx_tokens126 = backupTokens270
    h269 = backupH270
    E269 = backupE270
    for n,v in backupW270.items():
        rx_W[n] = v

print("elapsed seconds",round(time.time()-t270,1),"edits unchanged",edited())
with open("/work/notes.md","a") as f:
    f.write("\n## Run 270\nComputed finite-difference transformed feedback including current posterior/latent-token response; fitted feedback-penalized recurrent injection directions without gradients. No weights changed. Counts: 270 Python runs, 28 forwards.\n")
```
