"""Presentation-only normalization and contrast audit for physical error maps."""
import numpy as np
from matplotlib.colors import LogNorm,SymLogNorm


def visible_region(shape,X=None,Y=None,view=None):
    region=np.ones(shape,dtype=bool)
    if X is not None:
        X=np.asarray(X);Y=np.asarray(Y)
        region &= np.isfinite(X)&np.isfinite(Y)
        if view is not None:
            left,right,bottom,top=view
            region &= (X>=left)&(X<=right)&(Y>=bottom)&(Y<=top)
    return region


def contrast(image,norm,region):
    """Fraction of actually coloured, visible cells in the lowest 5% of colour."""
    values=np.asarray(image,dtype=np.float64)[region]
    values=values[np.isfinite(values)]
    if isinstance(norm,LogNorm):values=values[values>0]
    if not len(values):return None,True
    mapped=np.ma.masked_invalid(norm(values)).compressed()
    return float(np.mean(mapped<=.05)),bool(np.all(values==values[0]))


def check_contrast(audit):
    fraction=audit['lowest_five_percent_fraction']
    if not audit['constant'] and fraction is not None and fraction>.90:
        raise ValueError(f'More than 90% of visible map pixels occupy the lowest 5% of colour ({fraction:.1%})')


def error_colors(image,region,*,signed=False):
    """Use only positive magnitudes inside the view for percentile calibration."""
    image=np.asarray(image,dtype=np.float64)
    finite=np.isfinite(image);values=image[region&finite]
    if not signed and np.any(image[finite]<0):raise ValueError('Physical error map contains negative values')
    magnitude=np.abs(values) if signed else values
    positive=magnitude[magnitude>0]
    low,high=(np.percentile(positive,[1,99.5]) if len(positive) else (None,None))
    if signed:
        bound=float(high) if high is not None else 1.
        threshold=float(low) if low is not None else 1.
        norm=SymLogNorm(threshold,vmin=-bound,vmax=bound,base=10)
        data=np.ma.masked_invalid(image)
    else:
        vmin,vmax=(float(low),float(high)) if low is not None else (1.,10.)
        if vmin==vmax:vmin/=np.sqrt(10.);vmax*=np.sqrt(10.)
        norm=LogNorm(vmin=vmin,vmax=vmax)
        data=np.ma.masked_where(~finite|(image<=0),image)
    fraction,constant=contrast(image,norm,region)
    audit={'normalization':type(norm).__name__,'vmin':float(norm.vmin),'vmax':float(norm.vmax),
           'positive_p01':None if low is None else float(low),'positive_p995':None if high is None else float(high),
           'linthresh':float(norm.linthresh) if signed else None,
           'visible_cells':int(np.count_nonzero(region)),'positive_cells':int(len(positive)),
           'zero_cells':int(np.count_nonzero(values==0)),'nonfinite_cells':int(np.count_nonzero(region&~finite)),
           'lowest_five_percent_fraction':fraction,'constant':constant}
    notice=None
    if not len(values):notice='No finite error values in displayed region'
    elif not len(positive):notice='Zero bias in displayed region' if signed else 'Zero error in displayed region'
    elif low==high:notice='Constant central error range'
    return data,norm,audit,notice
