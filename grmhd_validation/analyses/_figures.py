"""Fixed production figures of all-start means and pointwise intervals.

Panel quantities and sizes adapted from Kevin Fortune/Sachin Venkatesh's
Fall26/eval/evaluate_forecast_ds.py, main@ebcc218. The legacy plots remain
available through the explicit compute API for source regression.
"""
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


def curve(ax,stats,paths,x=None,select=None,display=None):
    for path in paths:
        y=np.asarray(at(stats['mean'],path),dtype=np.float64)
        lo=np.asarray(at(stats['lower'],path),dtype=np.float64);hi=np.asarray(at(stats['upper'],path),dtype=np.float64)
        if select is not None:y=y[select];lo=lo[select];hi=hi[select]
        if display is not None:y=display(y);lo=display(lo);hi=display(hi)
        if y.ndim==0:y=y.reshape(1);lo=lo.reshape(1);hi=hi.reshape(1)
        if x is None:axis=np.arange(1,y.shape[-1]+1)
        else:
            axis=np.asarray(x)
            if y.ndim==2 and y.shape[0]==len(axis) and y.shape[-1]!=len(axis):y=y.T;lo=lo.T;hi=hi.T
        y=y.reshape(-1,y.shape[-1]);lo=lo.reshape(y.shape);hi=hi.reshape(y.shape)
        if len(axis)!=y.shape[-1]:raise ValueError('Figure coordinate and values disagree: '+str(path))
        label=' '.join(map(str,path));style='--' if any('pred' in str(part) or part=='model' for part in path) else '-'
        for index,(value,low,high) in enumerate(zip(y,lo,hi)):
            line=ax.plot(axis,value,style,label=label+(f' {index}' if len(y)>1 else ''))[0]
            ax.fill_between(axis,low,high,color=line.get_color(),alpha=.15)
    ax.grid(True,alpha=.25);ax.legend(fontsize=6)


def render(ctx,name,stats):
    if not FIGURES.get(name):return
    import matplotlib
    matplotlib.use(ctx.constants['figures']['backend'])
    import matplotlib.pyplot as plt
    mean=stats['mean'];log=lambda x:np.log10(np.maximum(x,ctx.constants['full']['epsilon']))
    sizes=ctx.constants['figures'].get('full_figsizes',SIZES)
    with plt.rc_context({'font.family':ctx.constants['figures']['font'],'font.size':ctx.constants['figures']['font_size']}):
        for variant,filename in enumerate(FIGURES[name].values()):
            panels=[];maps=[]
            if name=='error_correlation':
                for field in ('model','persistence'):maps.append((mean[field],field,None,None))
                layout=(1,2)
            elif name=='shell_profiles':
                for quantity in ('mass','energy','angular'):
                    panels.append(([('samples',i,quantity+'_'+side) for i in range(len(mean['samples'])) for side in ('pred','truth')],mean['r'],None,None))
                    panels.append(([('samples',i,quantity+'_error') for i in range(len(mean['samples']))],mean['r'],None,None))
                layout=(3,2)
            elif name=='mdot':
                panels=[([('samples',i,side) for i in range(len(mean['samples'])) for side in ('pred','truth')],mean['r'],None,None),
                        ([('samples',i,'relative_error') for i in range(len(mean['samples']))],None,None,None)];layout=(2,1)
            elif name=='accretion_diagnostics':
                for field in ('magnetic_flux','p_jet'):
                    panels.append(([('profiles_'+side,field) for side in ('pred','truth')],mean['r'],None,None))
                panels.append(([('headline_'+side,'phi_normalized') for side in ('pred','truth')],mean['report_steps'],None,None))
                layout=(1,3)
            elif name=='mean_profiles':
                for field in (['omega','rho_theta','beta','sigma'] if variant==0 else ['omega','beta','sigma']):
                    paths=[(field+'_'+side,) for side in ('pred','truth')]
                    if field=='omega':paths.append(('omega_keplerian',))
                    panels.append((paths,mean['theta'] if field=='rho_theta' else mean['r'],None,log if field in {'beta','sigma'} else None))
                layout=(2,2) if variant==0 else (1,3)
            elif name=='temporal_consistency':
                for field in ('mass','energy','angular'):
                    panels.append(([(field+'_drift_'+side,) for side in ('pred','truth')],mean['steps'],None,None))
                layout=(2,2)
            elif name in {'statistical_props','spectrum','turbulence_stats'}:
                for index,item in enumerate(mean['samples']):
                    fields={'statistical_props':('pdf_pred','pdf_truth'),'spectrum':('power_pred','power_truth'),'turbulence_stats':('s2_normalized_pred','s2_normalized_truth')}[name]
                    x=item.get('centers',item.get('modes',item.get('lags')))
                    panels.append(([('samples',index,field) for field in fields],x,None,None))
                if name=='spectrum':panels.append(([('rho_ratio',),('B1_ratio',)],mean['rho_modes'],None,None))
                layout=(2,4) if name=='statistical_props' else (1,3)
            elif name=='azimuthal_spectra':
                for channel in ('0','4','5'):
                    panels.append(([(side,step,channel) for side in ('pred','truth') for step in mean[side]],mean['modes'],None,None))
                for channel in ('0','4','5'):
                    panels.append(([('ratio',step,channel) for step in mean['ratio']],mean['modes'],None,None))
                layout=(2,3)
            elif name=='coherence':
                maps=[(np.asarray(mean['coherence'][channel]).T,f'coherence {ctx.channels[int(channel)]}',None,None) for channel in ('0','5')];layout=(1,2)
            elif name=='temporal_spectrum':
                for channel in range(8):panels.append(([(side+'_'+region,) for side in ('pred','truth') for region in ('disk','jet')],mean['frequencies'],channel,None))
                layout=(2,4)
            elif name=='conditional_error':
                for channel in range(8):panels.append(([('model',),('persistence',)],np.arange(1,len(mean['counts'])+1),(slice(None),channel),None))
                layout=(2,4)
            elif name=='spatial_error':
                for index,item in enumerate(mean['samples']):
                    for region in ('disk','jet'):panels.append(([('samples',index,region)],item['radius'],(slice(None),slice(0,5) if variant==0 else slice(5,8)),None))
                layout=(len(mean['samples']),2)
            elif name=='bias_analysis':
                r=ctx.geometry.r.detach().cpu().numpy()[:,:,0];t=ctx.geometry.theta.detach().cpu().numpy()[:,:,0];X=r*np.sin(t);Y=r*np.cos(t)
                for field in ('bias','rmse'):
                    for channel in (0,4,5,7):maps.append((mean[field][channel],f'{field} {ctx.channels[channel]}',X,Y))
                layout=(2,4)
            elif name=='meridional_slice':
                for fields in (('rho_truth','rho_pred','rho_error'),('B2_truth','B2_pred','B2_error')):
                    for field in fields:maps.append((log(np.asarray(mean[field])) if 'error' not in field else mean[field],field,mean['X'],mean['Y']))
                layout=(2,3)
            elif name=='rollout_evolution':
                for field in ('truth','pred','error'):
                    for item in mean['samples']:maps.append((log(np.asarray(item[field])) if field!='error' else item[field],f'{field} step {item["step"]}',mean['X'],mean['Y']))
                layout=(3,len(mean['samples']))
            else:raise ValueError('Missing fixed production figure: '+name)
            fig,axes=plt.subplots(*layout,figsize=sizes[name][variant],squeeze=False)
            try:
                for ax,definition in zip(axes.flat,panels):
                    paths,x,select,display=definition
                    curve(ax,stats,paths,x,select,display)
                    ax.set_title(' / '.join(map(str,paths[0][:-1] or paths[0])))
                for ax,(image,title,X,Y) in zip(axes.flat,maps):
                    if X is None:
                        if name=='coherence':
                            artist=ax.imshow(image,cmap='magma',vmin=0,vmax=1,origin='lower',aspect='auto',extent=[mean['steps'][0],mean['steps'][-1],0,len(image)])
                            ax.set_xlabel('Forecast step');ax.set_ylabel('Radial Fourier index')
                        else:
                            artist=ax.imshow(image,cmap='RdBu_r',vmin=-1,vmax=1)
                            ax.set_xticks(range(8),ctx.channels,rotation=45,ha='right');ax.set_yticks(range(8),ctx.channels)
                    else:artist=ax.pcolormesh(X,Y,image,shading='auto');ax.set_aspect('equal')
                    if name=='meridional_slice' and variant:
                        radius=(100,250)[variant-1];ax.set_xlim(0,radius);ax.set_ylim(-radius,radius)
                    ax.set_title(title);fig.colorbar(artist,ax=ax)
                for ax in list(axes.flat)[max(len(panels),len(maps)):]:ax.set_visible(False)
                fig.suptitle(f'{name}: {stats["n_starts"]} starts, mean and 95% CI')
                fig.tight_layout()
                with ctx.writer.open('figures/'+filename) as stream:
                    fig.savefig(stream,format='png',dpi=ctx.constants['figures']['dpi'],metadata={'Software':None})
            finally:plt.close(fig)
