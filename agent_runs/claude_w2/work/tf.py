
def teacher_forced(df, ids, W, chunk=500):
    outs=[]
    for i in range(0,len(ids),chunk):
        R=prep_rollout(df, ids[i:i+chunk]); res=run_rollout(R,W,[{}],n_paths=1,seed=0,t_hide=1e9,want_nll=True)
        A=R['A']; meas=np.where(np.isin(A.code.values,[CODE[c] for c in MEAS]))[0]
        key=meas.astype(np.int64)+0  # n = si (one arm one path) -> n*10**7+j
        n_=A.si.values[meas]
        pm=np.array([res['pm'].get(int(n*10**7+j),np.nan) for n,j in zip(n_,meas)])
        nl=np.array([res['nll'].get(int(n*10**7+j),np.nan) for n,j in zip(n_,meas)])
        o=pd.DataFrame({'stay_id':R['stays'][n_],'t':A.t.values[meas],'code':A.code.values[meas],'value':A.value.values[meas],'pm':pm,'nll':nl})
        outs.append(o)
    return pd.concat(outs,ignore_index=True)
