"""Full-figure integration for logarithmic physical error colours."""
import importlib
from unittest.mock import patch
import numpy as np
from matplotlib.colors import LogNorm,SymLogNorm
from test_full_analyses import FullFixture
from grmhd_validation.analyses._map_colors import visible_region,contrast


class ErrorMapPresentationTests(FullFixture):
    def test_every_physical_error_map_has_readable_log_colours(self):
        ctx=self.context();seen=[]
        r=ctx.geometry.r.numpy()[:,:,0];theta=ctx.geometry.theta.numpy()[:,:,0]
        X=r*np.sin(theta);Y=r*np.cos(theta)
        def inspect(fig,*args,**kwargs):
            for ax in fig.axes:
                for artist in ax.collections:
                    if not hasattr(artist,'_gval_error_map'):continue
                    audit=artist._gval_error_map;seen.append(audit)
                    self.assertIsInstance(artist.norm,(LogNorm,SymLogNorm))
                    label=artist.colorbar.ax.get_ylabel()
                    self.assertIn('code units',label);self.assertIn('log scale',label)
                    image=np.asarray(artist.get_array()).reshape(r.shape)
                    region=visible_region(r.shape,X,Y,audit['view'])
                    fraction,constant=contrast(image,artist.norm,region)
                    if not constant and fraction is not None:self.assertLessEqual(fraction,.90,ax.get_title())
                    if isinstance(artist.norm,SymLogNorm):self.assertEqual(artist.norm(0.),.5)
        with patch('matplotlib.figure.Figure.savefig',autospec=True,side_effect=inspect):
            for name in ('meridional_slice','bias_analysis','rollout_evolution'):
                importlib.import_module('grmhd_validation.analyses.'+name).run(ctx)
        self.assertEqual(len(seen),18)
        self.assertEqual(sum(s['normalization']=='LogNorm' for s in seen),14)
        self.assertEqual(sum(s['normalization']=='SymLogNorm' for s in seen),4)
        self.assertIn(None,[s['view'] for s in seen]);self.assertIn((0,250,-250,250),[s['view'] for s in seen])
        ctx.full.cleanup()
