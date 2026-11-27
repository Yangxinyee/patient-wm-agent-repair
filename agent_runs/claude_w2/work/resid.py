
def build_resid_tab(tf_, gg_):
    G=gg_.sort_values(['stay_id','h']).copy()
    gb=G.groupby('stay_id')
    G['nexp']=G['nor_l0']-G['e_nor']
    for k in range(1,6):
        G['pol_l%d'%k]=gb.ins_base.shift(k); G['ei_l%d'%k]=gb.e_ins.shift(k)
        G['en_l%d'%k]=gb.e_nor.shift(k); G['nexp_l%d'%k]=gb.nexp.shift(k)
    T=tf_[np.abs(tf_.t-np.round(tf_.t))<1e-6].copy(); T['h']=np.round(T.t).astype(int)
    M=T.merge(G,on=['stay_id','h'],how='left'); M['res']=M.value-M.pm
    return M
def ols_tab(M,ycol,xcols):
    d=M[[ycol]+xcols].dropna(); X=np.c_[np.ones(len(d)),d[xcols].values]; y=d[ycol].values
    b,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@b; s2=r@r/(len(y)-X.shape[1])
    se=np.sqrt(np.diag(np.linalg.inv(X.T@X))*s2)
    return pd.DataFrame({'b':b,'se':se},index=['const']+xcols).round(3), len(y), np.sqrt(s2)

def build_resid_tab2(tf_, gg_, K=6):
    G=gg_.sort_values(['stay_id','h'])[['stay_id','h','ins0','e_ins','nor0','e_nor','glu_now','map_pol']].copy()
    G['polm']=G.ins0-G.e_ins; G['nexp']=G.nor0-G.e_nor
    gb=G.groupby('stay_id')
    for k in range(0,K+1):
        G['d%d'%k]=gb.ins0.shift(k); G['p%d'%k]=gb.polm.shift(k); G['e%d'%k]=gb.e_ins.shift(k)
        G['n%d'%k]=gb.nor0.shift(k); G['x%d'%k]=gb.nexp.shift(k); G['u%d'%k]=gb.e_nor.shift(k)
    T=tf_[np.abs(tf_.t-np.round(tf_.t))<1e-6].copy(); T['h']=np.round(T.t).astype(int)
    M=T.merge(G,on=['stay_id','h'],how='left'); M['res']=M.value-M.pm
    return M
