"""All-start presentation adapted from Kevin Fortune/Sachin Venkatesh.

Source: Fall26/eval/evaluate_forecast_ds.py, main@ebcc218. Explicit panels
retain the source physics design; input statistics and result content are unchanged.
"""
import re
import numpy as np
from ._aggregate import at
from ._port_manifest import FIGURES

SIZES={'error_correlation':[(18,7)],'shell_profiles':[(12,10)],
       'temporal_consistency':[(12,9)],'accretion_diagnostics':[(18,5.5)],
       'mdot':[(8,7)],'mean_profiles':[(13,10),(18,5)],
       'statistical_props':[(20,8)],'spectrum':[(16,5)],
       'azimuthal_spectra':[(15,8.5)],'coherence':[(13,5)],
       'temporal_spectrum':[(20,9)],'turbulence_stats':[(15,5)],
       'conditional_error':[(20,9)],'bias_analysis':[(20,8.5)],
       'spatial_error':[(14,16.8),(14,16.8)],
       'meridional_slice':[(15,8.5)]*3,'rollout_evolution':[(16,11.5)]}
CHANNEL_LABELS=(r'$\rho$',r'$u_{\rm g}$',r'$\widetilde u^r$',r'$\widetilde u^\theta$',
                r'$\widetilde u^\varphi$',r'$B^r$',r'$B^\theta$',r'$B^\varphi$')
CHANNEL_COLORS=('#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#17becf')
ROLE_COLORS={'truth':'#1f77b4','pred':'#d62728','model':'#d62728','persistence':'#444444'}
ROLE_LABELS={'truth':'Truth','pred':'Model','model':'Model','persistence':'Persistence'}
STEP_COLORS={1:'#2ca02c',10:'#ff7f0e',25:'#d62728',50:'#9467bd',100:'#8c564b'}
R_LABEL=r'$r\;[R_g]$'
T_LABEL=r'$t\;[R_g/c]$'
X_LABEL=r'$r\sin\theta\;[R_g]$'
Y_LABEL=r'$r\cos\theta\;[R_g]$'


def step_label(ctx,step):
    time=float(step)*float(ctx.split.dt_dump)
    number=f'{time:,.0f}' if time.is_integer() else f'{time:,.3g}'
    return f'step {int(step)} ({number} '+r'$R_g/c$'+')'


def step_color(step):
    return STEP_COLORS.get(int(step),CHANNEL_COLORS[int(step)%len(CHANNEL_COLORS)])


def caption(ctx,count):
    return f'{count} starts · {ctx.split.split_id} / {ctx.partition} · Bands: 95% CIs'


def log_display(value):
    value=np.asarray(value,dtype=np.float64)
    with np.errstate(divide='ignore',invalid='ignore'):return np.where(value>0,np.log10(value),np.nan)


def line(ax,stats,path,x,label,*,color,style='-',select=None,display=None,band=True):
    """Exactly one reduced curve and one interval, never flattened field arrays."""
    values=[np.asarray(at(stats[field],path),dtype=np.float64) for field in ('mean','lower','upper')]
    if select is not None:values=[value[select] for value in values]
    if display is not None:values=[display(value) for value in values]
    y,lo,hi=(np.atleast_1d(value) for value in values)
    if y.ndim!=1 or y.shape!=lo.shape or y.shape!=hi.shape or np.asarray(x).shape!=y.shape:
        raise ValueError('Figure needs an explicitly selected one-dimensional curve: '+str(path))
    lo,hi=np.minimum(lo,hi),np.maximum(lo,hi)
    artist=ax.plot(x,y,style,color=color,label=label,linewidth=1.6,zorder=3)[0]
    artist.set_gid('mean:'+label)
    if band:
        interval=ax.fill_between(x,lo,hi,color=color,alpha=.10,linewidth=0,zorder=1)
        interval.set_gid('ci95:'+label)
    return artist


def pair(ax,stats,paths,x,*,select=None,display=None,step=None,ctx=None,multi=False,band=True):
    for role,path in paths:
        label=ROLE_LABELS[role]
        if step is not None:label+=' — '+step_label(ctx,step)
        line(ax,stats,path,x,label,color=step_color(step) if multi else ROLE_COLORS[role],
             style='-' if role=='truth' else '--',select=select,display=display,band=band)


def report_legend(ctx,ax,steps):
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color='.2',linestyle=style,label=label) for style,label in (('-','Truth'),('--','Model'))]
    handles.extend(Line2D([],[],color=step_color(step),label=step_label(ctx,step)) for step in steps)
    ax.legend(handles=handles,fontsize=7,ncol=2,framealpha=.85)


def panel(ax,title,xlabel,ylabel,*,xscale='linear',yscale='linear',legend=True):
    if xscale=='log':
        values=np.concatenate([np.asarray(artist.get_xdata(),dtype=float).ravel() for artist in ax.lines])
        values=values[np.isfinite(values)&(values>0)]
        if len(values):
            low,high=float(values.min()),float(values.max())
            ax.set_xlim((low,high) if high>low else (.9*low,1.1*high))
        else:ax.set_xlim(1.,10.)
    if yscale=='log' and not any(np.any(np.isfinite(artist.get_ydata())&(np.asarray(artist.get_ydata())>0)) for artist in ax.lines):
        # Undefined normalized curves (e.g. zero S2(1)) must remain undefined,
        # rather than acquiring fabricated positive values for a logarithm.
        ax.set_ylim(1.,10.)
        ax.text(.5,.5,'No finite positive values\non logarithmic scale',transform=ax.transAxes,ha='center',va='center',fontsize=9)
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel,xscale=xscale,yscale=yscale)
    if yscale=='symlog':
        pool=np.concatenate([np.asarray(artist.get_ydata(),dtype=float).ravel() for artist in ax.lines if (artist.get_gid() or '').startswith('mean:')])
        pool=np.abs(pool[np.isfinite(pool)&(pool!=0)])
        threshold=10**np.floor(np.log10(max(1e-4,float(np.percentile(pool,5))))) if len(pool) else .1
        ax.set_yscale('symlog',linthresh=threshold,linscale=1.)
    ax.grid(True,alpha=.23,which='both')
    if legend and ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=7,ncol=2 if len(ax.get_legend_handles_labels()[1])>4 else 1,framealpha=.85)


def radial(ctx,ax,r,*,full=False):
    from ._kevin_math import kerr_horizon_isco
    horizon,isco=kerr_horizon_isco(float(ctx.geometry.spin))
    left=max(horizon,float(np.min(r)));right=float(np.max(r)) if full else ctx.constants['active_disk']['r_max']
    ax.set_xlim(left,right)
    # Match the source's pre-plot radial cut: invisible outer-domain values
    # must not set the vertical scale of an inner-disk panel.
    pieces=[]
    for artist in ax.lines:
        if artist.get_label()=='_nolegend_':continue
        x=np.asarray(artist.get_xdata(),dtype=float);y=np.asarray(artist.get_ydata(),dtype=float)
        pieces.append(y[(x>=left)&(x<=right)])
    for artist in ax.collections:
        if not (artist.get_gid() or '').startswith('ci95:'):continue
        for path in artist.get_paths():
            xy=path.vertices;pieces.append(xy[(xy[:,0]>=left)&(xy[:,0]<=right),1])
    pool=np.concatenate(pieces) if pieces else np.array([])
    pool=pool[np.isfinite(pool)]
    if ax.get_yscale()=='log':pool=pool[pool>0]
    if len(pool):
        low,high=float(pool.min()),float(pool.max())
        if ax.get_yscale()=='log':ax.set_ylim(low/1.12,high*1.12)
        else:
            margin=.05*max(high-low,abs(high),abs(low),1e-12)
            ax.set_ylim(max(0.,low-margin) if low>=0 else low-margin,high+margin)
    ax.axvline(isco,color='.45',linestyle=':',linewidth=.8,label='_nolegend_')


def color_map(fig,ax,image,title,label,*,X=None,Y=None,limits=None,cmap='inferno',extent=None):
    if X is None:
        artist=ax.imshow(np.ma.masked_invalid(image),cmap=cmap,origin='lower',aspect='auto',extent=extent,
                         vmin=None if limits is None else limits[0],vmax=None if limits is None else limits[1])
    else:
        artist=ax.pcolormesh(X,Y,np.ma.masked_invalid(image),shading='auto',cmap=cmap,
                             vmin=None if limits is None else limits[0],vmax=None if limits is None else limits[1])
        ax.set(xlabel=X_LABEL,ylabel=Y_LABEL,aspect='equal')
    ax.set_title(title)
    fig.colorbar(artist,ax=ax,pad=.02,fraction=.046).set_label(label)


def limits(values,*,symmetric=False):
    pool=np.concatenate([np.asarray(value,dtype=np.float64).ravel() for value in values])
    pool=pool[np.isfinite(pool)]
    if not len(pool):return (0.,1.)
    if symmetric:
        bound=max(float(np.percentile(np.abs(pool),99.5)),np.finfo(float).eps)
        return -bound,bound
    lo,hi=np.percentile(pool,[1,99.5]);return (float(lo),float(hi) if hi>lo else float(lo)+1.)


def validate_figure(fig):
    """Reject raw result keys and unlabeled visible axes before writing a PNG."""
    forbidden=re.compile(r'\b\w+_(?:pred|truth|model|persistence)\b|headline_|samples\b|magnetic_flux|phi_normalized|rho_theta|s2_normalized|\bTRON\b|\butilde[123]\b')
    audit=[]
    for ax in fig.axes:
        if not ax.get_visible() or not ax.axison:continue
        if ax.get_label()=='<colorbar>':
            if not (ax.get_ylabel() or ax.get_xlabel()):raise ValueError('Unlabeled figure colour bar')
            continue
        if not all((ax.get_title(),ax.get_xlabel(),ax.get_ylabel())):raise ValueError('Unlabeled figure data axis')
        legend=ax.get_legend()
        labels=[ax.get_title(),ax.get_xlabel(),ax.get_ylabel()]
        if legend is not None:labels.extend(text.get_text() for text in legend.get_texts())
        if any(forbidden.search(label) for label in labels):raise ValueError('Placeholder figure label: '+str(labels))
        audit.append({'title':ax.get_title(),'xlabel':ax.get_xlabel(),'ylabel':ax.get_ylabel(),
                      'xscale':ax.get_xscale(),'yscale':ax.get_yscale(),
                      'legend':[] if legend is None else [text.get_text() for text in legend.get_texts()],
                      'mean_curves':sum((artist.get_gid() or '').startswith('mean:') for artist in ax.lines),
                      'ci_bands':sum((artist.get_gid() or '').startswith('ci95:') for artist in ax.collections)})
    return {'panels':audit,'caption':[text.get_text() for text in fig.texts if text is not fig._suptitle]}


def build_figure(ctx,name,stats,variant=0):
    import matplotlib.pyplot as plt
    m=stats['mean'];sizes=ctx.constants['figures'].get('full_figsizes',SIZES)
    layouts={'error_correlation':(1,2),'shell_profiles':(3,2),'temporal_consistency':(2,2),
             'accretion_diagnostics':(1,3),'mdot':(2,1),'mean_profiles':(2,2) if variant==0 else (1,3),
             'statistical_props':(2,4),'spectrum':(1,3),'azimuthal_spectra':(2,3),'coherence':(1,2),
             'temporal_spectrum':(2,4),'turbulence_stats':(1,3),'conditional_error':(2,4),
             'bias_analysis':(2,4),'spatial_error':(len(m.get('samples',[])),2),
             'meridional_slice':(2,3),'rollout_evolution':(3,len(m.get('samples',[])))}
    fig,axes=plt.subplots(*layouts[name],figsize=sizes[name][variant],squeeze=False)
    flat=list(axes.flat);title=''
    if name=='error_correlation':
        title='Cross-Channel Error Correlation'
        for ax,field in zip(flat,('model','persistence')):
            color_map(fig,ax,np.asarray(m[field]),ROLE_LABELS[field]+' Error Correlation','Pearson correlation',limits=(-1,1),cmap='RdBu_r')
            ax.set(xlabel='Channel',ylabel='Channel')
            ax.set_xticks(range(8),CHANNEL_LABELS,rotation=45,ha='right');ax.set_yticks(range(8),CHANNEL_LABELS)
            for i in range(8):
                for j in range(8):
                    value=m[field][i][j];ax.text(j,i,'n/a' if value is None or not np.isfinite(value) else f'{value:.2f}',ha='center',va='center',fontsize=8)
    elif name=='shell_profiles':
        title='Conserved Shell Profiles';r=np.asarray(m['r'])
        for row,(field,label) in enumerate((('mass',r'$M(r)$'),('energy',r'$-\mathcal{E}(r)$ (rest-mass-subtracted)'),('angular',r'$L(r)$'))):
            for i,sample in enumerate(m['samples']):
                h=sample['step']
                pair(axes[row,0],stats,[(side,('samples',i,field+'_'+side)) for side in ('truth','pred')],r,
                     display=(lambda v:-v) if field=='energy' else None,step=h,ctx=ctx,multi=True,band=i==len(m['samples'])-1)
                line(axes[row,1],stats,('samples',i,field+'_error'),r,step_label(ctx,h),color=step_color(h),band=i==len(m['samples'])-1)
            panel(axes[row,0],label+' Profile',R_LABEL,label+' [code units]',xscale='log',yscale='log')
            panel(axes[row,1],label+' Error',R_LABEL,'Signed-log absolute error',xscale='log',yscale='log')
            for ax in axes[row]:radial(ctx,ax,r,full=row==0)
            report_legend(ctx,axes[row,0],[sample['step'] for sample in m['samples']])
    elif name=='mdot':
        title='Accretion Rate Profiles';r=np.asarray(m['r']);horizon=1+np.sqrt(max(0.,1-float(ctx.geometry.spin)**2))
        active=(r>=horizon)&(r<=ctx.constants['active_disk']['r_max'])
        for i,sample in enumerate(m['samples']):
            h=sample['step']
            pair(flat[0],stats,[(side,('samples',i,side)) for side in ('truth','pred')],r,step=h,ctx=ctx,multi=True,band=i==len(m['samples'])-1)
            line(flat[1],stats,('samples',i,'relative_error'),r[active],step_label(ctx,h),color=step_color(h),band=i==len(m['samples'])-1)
        panel(flat[0],'Accretion Rate Profiles',R_LABEL,r'$\dot M(r)$ [code units]',xscale='log',yscale='symlog')
        panel(flat[1],'Accretion Rate Relative Error',R_LABEL,'Relative error',xscale='log',yscale='log')
        for ax in flat:radial(ctx,ax,r)
        report_legend(ctx,flat[0],[sample['step'] for sample in m['samples']])
    elif name=='accretion_diagnostics':
        title='Accretion & Jet Diagnostics';r=np.asarray(m['r']);steps=m['report_steps'];last=len(steps)-1
        for ax,field,heading,ylabel in ((flat[0],'magnetic_flux','(a) Magnetic Flux Threading',r'$\Phi(r)=\int |B^r|\sqrt{-g}\,d\theta\,d\varphi$'+'\n[code units]'),
                                      (flat[1],'p_jet','(b) Blandford–Znajek Jet Power (signed net flux)',r'$P_{\rm jet}(r)=-\int_{\rm funnel}(T^r{}_t+\rho u^r)\sqrt{-g}\,d\theta\,d\varphi$'+'\n[code units; rest mass removed]')):
            for i,step in enumerate(steps):
                pair(ax,stats,[(side,('profiles_'+side,field)) for side in ('truth','pred')],r,select=i,step=step,ctx=ctx,multi=True,band=i==last)
            panel(ax,heading,R_LABEL,ylabel,xscale='log',yscale='log' if field=='magnetic_flux' else 'symlog');radial(ctx,ax,r)
            report_legend(ctx,ax,steps)
        shell=m['shells']['resolved'];where=f'resolved shell, r={shell["radius_rg"]:.2f} '+r'$R_g$'
        pair(flat[2],stats,[(side,('headline_'+side,'phi_normalized')) for side in ('truth','pred')],np.asarray(steps)*ctx.split.dt_dump)
        panel(flat[2],'(c) Dimensionless Horizon Flux '+r'$\phi_{\rm BH}$'+'\n'+where,T_LABEL,r'$\phi_{\rm BH}=\Phi/(2\sqrt{\dot M})$ [dimensionless]')
        flat[2].set_xticks(np.asarray(steps)*ctx.split.dt_dump,[step_label(ctx,h) for h in steps],rotation=25,ha='right',fontsize=7)
    elif name=='mean_profiles':
        title='Mean Radial Physics Profiles'+(' — Full Domain' if variant else '')
        fields=('omega','rho_theta','beta','sigma') if variant==0 else ('omega','beta','sigma')
        for ax,field in zip(flat,fields):
            if field=='rho_theta':
                for i,radius in enumerate(ctx.constants['full']['mean_profile_radii']):
                    normal=float(np.nanmax(np.asarray(m['rho_theta_truth'])[i]));normal=normal if normal>0 else 1.
                    for side in ('truth','pred'):
                        line(ax,stats,(field+'_'+side,),np.degrees(m['theta']),ROLE_LABELS[side]+f' r={radius:g} '+r'$R_g$',color=CHANNEL_COLORS[i],
                             style='-' if side=='truth' else '--',select=i,display=lambda y,n=normal:y/n)
                panel(ax,'Vertical Disk Structure (phi-averaged)',r'$\theta$ [degrees]',r'$\rho(\theta)/\rho_{\rm truth,max}$',yscale='log');ax.set_xlim(0,180)
            else:
                pair(ax,stats,[(side,(field+'_'+side,)) for side in ('truth','pred')],m['r'],display=log_display if field in ('beta','sigma') else None)
                if field=='omega':ax.plot(m['r'],m['omega_keplerian'],':',color='.25',label='Kerr Keplerian',linewidth=1.2)
                heading={'omega':'Rotation Curve','beta':'Plasma Beta Profile','sigma':'Magnetization Profile'}[field]
                ylabel={'omega':r'$\Omega=u^\varphi/u^t\;[c/R_g]$','beta':r'$\log_{10}\beta=\log_{10}(p_{\rm gas}/p_{\rm mag})$','sigma':r'$\log_{10}\sigma=\log_{10}(b^2/\rho)$'}[field]
                panel(ax,heading+' (equatorial band)',R_LABEL,ylabel,xscale='log',yscale='log' if field=='omega' else 'linear')
                radial(ctx,ax,np.asarray(m['r']),full=bool(variant))
    elif name=='temporal_consistency':
        title='Total Conservation Drift'
        for ax,field,heading in zip(flat,('mass','energy','angular'),('Mass M_total','Energy E_total','Angular Momentum L_total')):
            pair(ax,stats,[(side,(field+'_drift_'+side,)) for side in ('truth','pred')],np.asarray(m['steps'])*ctx.split.dt_dump)
            panel(ax,heading,T_LABEL,'Absolute drift / initial scale [dimensionless]',yscale='log')
        flat[3].axis('off')
        text='Final mean drift (Model / Truth)\n\n'+'\n'.join(f'{heading}: {m[field+"_drift_pred"][-1]:.3e} / {m[field+"_drift_truth"][-1]:.3e}' for field,heading in (('mass','Mass'),('energy','Energy'),('angular','Angular momentum')))
        flat[3].text(.05,.6,text,transform=flat[3].transAxes,fontsize=10,va='center')
    elif name in ('statistical_props','spectrum','turbulence_stats'):
        title={'statistical_props':'State Distributions on Shared Truth Bins','spectrum':'Windowed Radial Power Spectra','turbulence_stats':'Turbulent Structure Functions'}[name]
        fields={'statistical_props':'pdf','spectrum':'power','turbulence_stats':'s2_normalized'}[name]
        for i,(ax,item) in enumerate(zip(flat,m['samples'])):
            channel=int(item['channel']);x=item.get('centers',item.get('modes',item.get('lags')))
            pair(ax,stats,[(side,('samples',i,fields+'_'+side)) for side in ('truth','pred')],x)
            suffix=' — '+step_label(ctx,m['step'])
            if name=='statistical_props':
                panel(ax,CHANNEL_LABELS[channel]+f': JS={item["js"]:.3g}'+suffix,CHANNEL_LABELS[channel]+' [code units]','PDF [inverse code units]')
            elif name=='spectrum':
                panel(ax,CHANNEL_LABELS[channel]+' Radial Spectrum'+suffix,r'Radial mode index $|k|$',r'Power $|\mathcal{F}[S]|^2$ [code units$^2$]',xscale='log',yscale='log');ax.set_xlim(left=1)
            else:
                panel(ax,CHANNEL_LABELS[channel]+suffix,r'Radial lag $\Delta\ln r$ [cells]',r'$S_2/S_2(1)$ [dimensionless]',xscale='log',yscale='log')
        if name=='spectrum':
            for channel,field in ((0,'rho'),(5,'B1')):line(flat[2],stats,(field+'_ratio',),m[field+'_modes'],CHANNEL_LABELS[channel],color=CHANNEL_COLORS[channel])
            flat[2].axhline(1,color='.3',linestyle=':',linewidth=.8)
            panel(flat[2],'Scale-Resolved Power Ratio',r'Radial mode index $|k|$',r'$E_{\rm model}/E_{\rm truth}$ [dimensionless]',xscale='log',yscale='log');flat[2].set_xlim(left=1)
    elif name=='azimuthal_spectra':
        title='Azimuthal Mode Spectra and Multi-Horizon Drift';steps=m['steps'];representative=50 if 50 in steps else steps[-1]
        for col,channel in enumerate((0,4,5)):
            key=str(channel);h=str(int(representative))
            pair(axes[0,col],stats,[(side,(side,h,key)) for side in ('truth','pred')],m['modes'])
            panel(axes[0,col],CHANNEL_LABELS[channel]+' Spectrum — '+step_label(ctx,representative),r'Azimuthal wavenumber $m$',r'Power $|F_m|^2$ [code units$^2$]',xscale='log',yscale='log');axes[0,col].set_xlim(left=1)
            for step in steps:line(axes[1,col],stats,('ratio',str(int(step)),key),m['modes'],step_label(ctx,step),color=step_color(step),band=step==steps[-1])
            axes[1,col].axhline(1,color='.3',linestyle=':',linewidth=.8)
            panel(axes[1,col],CHANNEL_LABELS[channel]+' Power Ratio',r'Azimuthal wavenumber $m$','Power ratio (Model / Truth)',xscale='log');axes[1,col].set_xlim(left=1)
    elif name=='coherence':
        title='Spectral Coherence'
        for ax,channel in zip(flat,(0,5)):
            image=np.asarray(m['coherence'][str(channel)],dtype=np.float64).T
            time=np.asarray(m['steps'])*ctx.split.dt_dump
            color_map(fig,ax,image,'Coherence '+r'$C(k,t)$'+' — '+CHANNEL_LABELS[channel],'Coherence [dimensionless]',limits=(0,1),cmap='magma',extent=[time[0],time[-1],0,image.shape[0]-1])
            ax.set(xlabel=T_LABEL,ylabel=r'Radial wavenumber $k$')
    elif name=='temporal_spectrum':
        title='Temporal Power Spectra — Turbulent Disk and Jet'
        for channel,ax in enumerate(flat):
            for region in ('disk','jet'):
                for side in ('truth','pred'):
                    color=ROLE_COLORS[side] if region=='disk' else ('#17becf' if side=='truth' else '#ff7f0e')
                    line(ax,stats,(side+'_'+region,),m['frequencies'],ROLE_LABELS[side]+' '+region.title(),color=color,style=('-' if side=='truth' else '--') if region=='disk' else (':' if side=='truth' else '-.'),select=channel)
            panel(ax,CHANNEL_LABELS[channel],r'$f\;[c/R_g]$',r'PSD $P(f)$ [code units$^2\,R_g/c$]',xscale='log',yscale='log')
            frequency=np.asarray(m['frequencies']);ax.set_xlim(left=frequency[frequency>0][0])
    elif name=='conditional_error':
        title='Error Conditional on Truth Magnitude'
        for channel,ax in enumerate(flat):
            for role in ('model','persistence'):
                line(ax,stats,(role,),np.arange(1,len(m['counts'])+1),ROLE_LABELS[role],color=ROLE_COLORS[role],style='-' if role=='model' else '--',select=(slice(None),channel))
            panel(ax,CHANNEL_LABELS[channel],'Truth magnitude quantile (low → high)' if channel<2 else 'Truth absolute-magnitude quantile (weak → strong)','MAE [code units]')
            ax.set_xticks(np.arange(1,len(m['counts'])+1))
    elif name=='spatial_error':
        title='Spatial Error Profiles — '+('Fluid Channels' if variant==0 else 'Magnetic Channels')
        for i,item in enumerate(m['samples']):
            for col,(region,heading) in enumerate((('disk','Disk Core'),('jet','Jet Funnel'))):
                for channel in (range(5) if variant==0 else range(5,8)):
                    line(axes[i,col],stats,('samples',i,region),item['radius'],CHANNEL_LABELS[channel],color=CHANNEL_COLORS[channel],select=(slice(None),channel))
                panel(axes[i,col],heading+' — '+step_label(ctx,item['step']),R_LABEL,'Relative L1 error',xscale='log',yscale='log')
    elif name=='bias_analysis':
        title='Spatial Error Decomposition — Bias & RMSE (phi-averaged)'
        r=ctx.geometry.r.detach().cpu().numpy()[:,:,0];theta=ctx.geometry.theta.detach().cpu().numpy()[:,:,0];X=r*np.sin(theta);Y=r*np.cos(theta)
        for row,field in enumerate(('bias','rmse')):
            for col,channel in enumerate((0,4,5,7)):
                image=np.asarray(m[field][channel],dtype=np.float64)
                color_map(fig,axes[row,col],image,CHANNEL_LABELS[channel]+(' Bias (Signed Error)' if field=='bias' else ' RMSE'),'Signed error [code units]' if field=='bias' else 'RMSE [code units]',X=X,Y=Y,limits=limits([image],symmetric=field=='bias'),cmap='RdBu_r' if field=='bias' else 'magma')
                axes[row,col].set(xlim=(0,100),ylim=(-100,100))
    elif name=='meridional_slice':
        title='Meridional Slices — '+step_label(ctx,m['step'])
        for row,(field,quantity) in enumerate((('rho',r'$\rho$'),('B2',r'$B^2$'))):
            images=[log_display(m[field+'_'+side]) for side in ('truth','pred')];scale=limits(images)
            for col,(side,image) in enumerate(zip(('truth','pred'),images)):
                color_map(fig,axes[row,col],image,quantity+' '+ROLE_LABELS[side],r'$\log_{10}$ '+quantity+' [code units]',X=m['X'],Y=m['Y'],limits=scale)
            image=np.asarray(m[field+'_error'],dtype=np.float64)
            color_map(fig,axes[row,2],image,quantity+' Absolute Error',quantity+' absolute error [code units]',X=m['X'],Y=m['Y'],limits=(0,max(limits([image])[1],np.finfo(float).eps)),cmap='viridis')
        if variant:
            radius=(100,250)[variant-1]
            for ax in flat:ax.set(xlim=(0,radius),ylim=(-radius,radius))
    elif name=='rollout_evolution':
        title='Meridional Density Rollout Evolution'
        for col,item in enumerate(m['samples']):
            images=[log_display(item[side]) for side in ('truth','pred')];scale=limits(images)
            for row,(side,image) in enumerate(zip(('truth','pred'),images)):
                color_map(fig,axes[row,col],image,ROLE_LABELS[side]+' — '+step_label(ctx,item['step']),r'$\log_{10}\rho$ [code units]',X=m['X'],Y=m['Y'],limits=scale)
            image=np.asarray(item['error'],dtype=np.float64)
            color_map(fig,axes[2,col],image,'Absolute Error — '+step_label(ctx,item['step']),r'$|\Delta\rho|$ [code units]',X=m['X'],Y=m['Y'],limits=(0,max(limits([image])[1],np.finfo(float).eps)),cmap='viridis')
        for ax in flat:ax.set(xlim=(0,100),ylim=(-100,100))
    else:
        plt.close(fig);raise ValueError('Missing fixed production figure: '+name)
    fig.suptitle(title,fontsize=13)
    text=caption(ctx,stats['n_starts'])
    if name in ('shell_profiles','mdot','accretion_diagnostics','azimuthal_spectra'):
        steps=m.get('report_steps',m.get('steps',[sample['step'] for sample in m.get('samples',[])]))
        text=f'Lines: mean over {stats["n_starts"]} starts at each report step. Bands: 95% CI at step {int(steps[-1])} only. Full CIs for every step are in metrics/{name}.json.\n'
        text+=f'{ctx.split.split_id} / {ctx.partition}'
        if name=='accretion_diagnostics':text+=' · '+r'$\phi_{\rm BH}$'+' time series: a 95% CI band for each role over all steps.'
    elif name in ('error_correlation','coherence','bias_analysis','meridional_slice','rollout_evolution'):
        text=f'Maps: mean over {stats["n_starts"]} starts · {ctx.split.split_id} / {ctx.partition} · Pointwise 95% CIs: metrics/{name}.json'
    fig.text(.5,.012,text,ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,.94))
    validate_figure(fig)
    return fig


def render(ctx,name,stats):
    if not FIGURES.get(name):return
    import matplotlib
    matplotlib.use(ctx.constants['figures']['backend'])
    import matplotlib.pyplot as plt
    audit={}
    with plt.rc_context({'font.family':ctx.constants['figures']['font'],'font.size':ctx.constants['figures']['font_size']}):
        for variant,filename in enumerate(FIGURES[name].values()):
            fig=build_figure(ctx,name,stats,variant)
            fig.set_label(filename)
            try:
                audit[filename]=validate_figure(fig)
                with ctx.writer.open('figures/'+filename) as stream:
                    fig.savefig(stream,format='png',dpi=ctx.constants['figures']['dpi'],metadata={'Software':None})
            finally:plt.close(fig)
    return audit
