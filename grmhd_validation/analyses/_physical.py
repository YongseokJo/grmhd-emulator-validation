"""Float64 NumPy versions of the physical evaluation kernels.

Source formulas: Kevin Fortune/Sachin, main@ebcc218,
Fall26/eval/evaluate_forecast_ds.py and Fall26/common/normalizer.py/physics_loss.py.
No model decode clamps are applied to physical predictions during scoring.
The original float32 paths remain separately callable for source regression.
"""
import numpy as np


def array(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.ascontiguousarray(value,dtype=np.float64)


def batched(value):
    value=array(value)
    return value[None] if value.ndim==4 else value


def gamma(utilde,gcov):
    u=array(utilde);g=array(gcov)
    return np.sqrt(1+np.maximum(np.einsum('bi...,ij...,bj...->b...',u,g[1:,1:],u),0))[:,None]


def ucon(utilde,gcov,gcon):
    u=array(utilde);g=array(gcon)
    u0=gamma(u,gcov)*np.sqrt(-g[0,0])
    ui=u+u0*np.stack([g[0,i]/g[0,0] for i in (1,2,3)])[None]
    return np.concatenate((u0,ui),axis=1)


def divergence(b1,b2,b3,gdet,dx):
    fields=[array(gdet)*array(b) for b in (b1,b2,b3)]
    derivatives=[]
    for axis,(field,spacing,mode) in enumerate(zip(fields,dx,('edge','edge','wrap')),start=2):
        pad=[(0,0)]*field.ndim;pad[axis]=(1,1)
        padded=np.pad(field,pad,mode=mode)
        hi=[slice(None)]*field.ndim;lo=hi.copy();hi[axis]=slice(2,None);lo[axis]=slice(None,-2)
        derivatives.append((padded[tuple(hi)]-padded[tuple(lo)])/(2*spacing))
    return sum(derivatives)/(array(gdet)+1e-20)


def mass(raw,geometry):
    x=batched(raw)
    velocity=ucon(x[:,2:5],geometry.gcov,geometry.gcon)
    return np.sum(x[:,0:1]*velocity[:,0:1]*array(geometry.gdet),axis=(3,4),dtype=np.float64)


def conserved(raw,geometry):
    x=batched(raw);rho=x[:,0:1];ug=x[:,1:2]
    g=array(geometry.gcov);gi=array(geometry.gcon);gd=array(geometry.gdet)
    velocity=ucon(x[:,2:5],g,gi);u0=velocity[:,0:1]
    alpha=1/np.sqrt(-gi[0,0]+1e-300);lorentz=alpha*u0
    pressure=(float(geometry.eos_gamma)-1)*ug;enthalpy=rho+ug+pressure
    ucov=np.einsum('bi...,ij...->bj...',velocity,g)
    Bcon=np.concatenate((np.zeros_like(u0),alpha*x[:,5:8]),axis=1)
    dot=np.sum(ucov*Bcon,axis=1,keepdims=True)
    bcon=(Bcon+velocity*dot)/lorentz
    bcov=np.einsum('bi...,ij...->bj...',bcon,g)
    ndot=-alpha*bcon[:,0:1];bsq=np.sum(bcov*bcon,axis=1,keepdims=True)
    energy=(lorentz*(enthalpy+bsq)*ucov[:,0:1]+(pressure+bsq/2)*alpha+ndot*bcov[:,0:1])/alpha+rho*u0
    angular=(lorentz*(enthalpy+bsq)*ucov[:,3:4]+ndot*bcov[:,3:4])/alpha
    n_ang=x.shape[3]*x.shape[4]
    return tuple(np.sum(value*gd,axis=(3,4),dtype=np.float64)/n_ang for value in (energy,angular))


def radial_spectrum(raw,channel):
    field=batched(raw)[:,channel:channel+1]
    field=field-np.mean(field,axis=(-3,-2,-1),keepdims=True)
    nr,nt=field.shape[-3:-1]
    # torch.hann_window's original periodic=True convention, in float64.
    window=np.hanning(nr+1)[:-1,None,None]*np.hanning(nt+1)[None,:-1,None]
    fourier=np.fft.rfftn(np.ascontiguousarray(field*window),axes=(-3,-2,-1))
    full=np.mean(np.abs(fourier)**2,axis=(-2,-1))[0,0]
    half=nr//2;folded=np.zeros(half+1,dtype=np.float64);folded[0]=full[0]
    for k in range(1,half+1):folded[k]=full[k] if k==nr-k else full[k]+full[nr-k]
    return np.arange(half+1),folded


def active_mask(geo,constants):
    return (geo['r']<=constants['active_disk']['r_max']) & (np.abs(geo['h']-np.pi/2)<constants['active_disk']['theta_halfwidth'])


def normalized_divergence(raw,geometry,constants,geo=None,return_global=False):
    x=batched(raw);gd=array(geometry.gdet)
    div=divergence(x[:,5:6],x[:,6:7],x[:,7:8],gd,array(geometry.dx))
    magnitude=np.sqrt(np.sum(x[:,5:8]**2,axis=1,keepdims=True))
    denominator=np.maximum(magnitude,.001*np.median(magnitude))+1e-12
    field=np.abs(div)*np.cbrt(gd*np.prod(array(geometry.dx)))/denominator
    global_value=float(np.mean(field));value=global_value
    if geo is not None:
        trimmed={key:geo[key][1:-1,1:-1,:] for key in ('r','h')}
        mask=active_mask(trimmed,constants) & (x[0,0,1:-1,1:-1,:]>100*constants['loss_space']['floor_rho'])
        if mask.any():value=float(np.mean(field[0,0,1:-1,1:-1,:][mask]))
    return (value,global_value) if return_global else value


def flux_residual(first,second,geometry,constants,geo,relative,n_cells,seed):
    a=batched(first);b=batched(second);gd=array(geometry.gdet)
    ua=ucon(a[:,2:5],geometry.gcov,geometry.gcon)
    ub=ucon(b[:,2:5],geometry.gcov,geometry.gcon)
    ma=a[:,0:1]*ua[:,0:1]*gd;mb=b[:,0:1]*ub[:,0:1]*gd
    flux=[]
    for x,u in ((a,ua),(b,ub)):
        flux.append(divergence(*(x[:,0:1]*u[:,i:i+1] for i in (1,2,3)),gd,array(geometry.dx))*gd)
    residual=mb-ma+float(geometry.dt_dump)*(flux[0]+flux[1])/2
    trimmed={key:geo[key][1:-1,1:-1,:] for key in ('r','h')}
    mask=active_mask(trimmed,constants) & (b[0,0,1:-1,1:-1,:]>100*constants['loss_space']['floor_rho'])
    r=residual[0,0,1:-1,1:-1,:];m=mb[0,0,1:-1,1:-1,:]
    r,m=(r[mask],m[mask]) if mask.any() else (r.ravel(),m.ravel())
    if n_cells is not None and len(r)>n_cells:
        indices=np.random.default_rng(seed).choice(len(r),size=n_cells,replace=False)
        r=r[indices];m=m[indices]
    return float(np.mean(np.abs(r)/(np.abs(m)+1e-10))) if relative else float(np.mean(np.abs(r)))
