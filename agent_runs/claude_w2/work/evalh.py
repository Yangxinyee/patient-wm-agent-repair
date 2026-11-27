
def eval_arms(res, arms, R, K=None, drug='insulin', outcome='glucose_finger'):
    K=Kins if K is None else K
    DG=dose_stack(R, arms, drug); outl=[]
    key={'insulin':'ins','norepinephrine':'ne'}[drug]
    for ai in range(1,len(arms)):
        t0=arms[ai].get(key+'_from',12.0)
        rows,reached,o=readout(R,res,arms,drug,outcome,t0=t0,lo=int(t0)+1,hi=24)
        sh=np.nanmean(o[ai]-o[0],1)[reached]; tr_=truth_rows(R,DG,ai,rows[reached],K)
        lag=np.round(R['A'].t.values[rows[reached]]).astype(int)-int(t0)
        prof=[(np.nanmean(sh[lag==l]), np.mean(tr_[lag==l])) for l in [1,3,6,9,12] if (lag==l).any()]
        outl.append((str({k:v for k,v in arms[ai].items() if not isinstance(v,np.ndarray)}), round(float(np.nanmean(sh)),2), round(float(np.mean(tr_)),2), len(sh), [(round(a,1),round(b,1)) for a,b in prof]))
    return outl
