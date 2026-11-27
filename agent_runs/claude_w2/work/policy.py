
def build_policy_table(df):
    g=df[df.event.isin(['glucose_finger','glucose_lab'])&(np.abs(df.hour-np.round(df.hour))<1e-6)][['stay_id','hour','event','value']]
    gp=g.pivot_table(index=['stay_id','hour'],columns='event',values='value',aggfunc='first')
    gsel=gp['glucose_finger'].where(gp['glucose_finger'].notna(),gp.get('glucose_lab'))
    ins=df[df.event=='insulin'][['stay_id','hour','rate']].copy(); ins['h']=np.floor(ins.hour+1e-6)
    ir=ins.groupby(['stay_id','h']).rate.sum()
    G=pd.DataFrame({'g':gsel}).reset_index(); G['rate']=ir.reindex(pd.MultiIndex.from_arrays([G.stay_id,G.hour])).fillna(0).values
    G['gi']=np.clip(np.round(G.g).astype(int),0,600)
    tab=G.groupby('gi').rate.agg(['mean','count'])
    return G,tab
def make_pf(tab):
    xs=np.arange(0,601); m=tab['mean'].reindex(xs); c=tab['count'].reindex(xs).fillna(0)
    # smooth within +-3 mg/dL weighted by counts, but keep step at 150
    num=(m.fillna(0)*c).values; den=c.values; out=np.zeros(601)
    for i in xs:
        lo,hi=max(0,i-3),min(600,i+3)
        if i<150: hi=min(hi,149)
        else: lo=max(lo,150)
        out[i]=num[lo:hi+1].sum()/max(den[lo:hi+1].sum(),1)
    out[:150]=0.0
    def pf(x):
        x=np.asarray(x,dtype=float); return np.interp(x,xs,out)
    return pf,out

def make_pf(tab,minc=30):
    xs=np.arange(0,601); m=tab['mean'].reindex(xs); c=tab['count'].reindex(xs).fillna(0)
    m[c<minc]=np.nan; m[:150]=0.0
    m=m.ffill().bfill().values; m[:150]=0.0
    def pf(x):
        x=np.asarray(x,dtype=float); return np.interp(x,xs,m)
    return pf,m

def make_pf(tab,minc=30):
    xs=np.arange(0,601); m=tab['mean'].reindex(xs).copy(); c=tab['count'].reindex(xs).fillna(0)
    m[c.values<minc]=np.nan; m.iloc[:150]=0.0
    m=m.ffill().bfill().to_numpy(copy=True); m[:150]=0.0
    def pf(x):
        x=np.asarray(x,dtype=float); return np.interp(x,xs,m)
    return pf,m

def make_pf_band(tab,edges=(150,180,220,250,260,300),minc=200):
    xs=np.arange(0,601); m=tab['mean'].reindex(xs); c=tab['count'].reindex(xs).fillna(0).values; mv=np.nan_to_num(m.values)
    out=np.zeros(601); E=list(edges)+[601]
    for i in range(150,601):
        b=max(j for j in range(len(E)-1) if E[j]<=i); lo_b,hi_b=E[b],E[b+1]-1
        w=0
        while True:
            lo,hi=max(lo_b,i-w),min(hi_b,i+w)
            if c[lo:hi+1].sum()>=minc or (lo==lo_b and hi==hi_b): break
            w+=1
        out[i]=(mv[lo:hi+1]*c[lo:hi+1]).sum()/max(c[lo:hi+1].sum(),1)
    def pf(x):
        x=np.asarray(x,dtype=float); return np.interp(x,xs,out)
    return pf,out
