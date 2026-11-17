"""Visible-region error colours and a regression for horizon-dominated maps."""
import importlib
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.colors import LogNorm,Normalize,SymLogNorm

from grmhd_validation.analyses._map_colors import visible_region,error_colors,contrast,check_contrast
from test_full_analyses import FullFixture


class ErrorColourOracles(unittest.TestCase):
    def test_limits_use_positive_pixels_inside_the_actual_view(self):
        X,Y=np.meshgrid([1.,2.,200.],[1.,2.,200.])
        image=np.full((3,3),1e50);image[:2,:2]=[[1.,10.],[100.,1000.]];before=image.copy()
        region=visible_region(image.shape,X,Y,(0,100,-100,100))
        _,norm,audit,_=error_colors(image,region)
        self.assertIsInstance(norm,LogNorm)
        self.assertAlmostEqual(norm.vmin,1.27);self.assertAlmostEqual(norm.vmax,986.5)
        self.assertEqual(audit['visible_cells'],4)
        np.testing.assert_array_equal(image,before)

    def test_signed_bias_is_symmetric_about_zero(self):
        image=np.array([[-1000.,-10.,0.,10.,1000.]])
        _,norm,audit,_=error_colors(image,np.ones_like(image,dtype=bool),signed=True)
        self.assertIsInstance(norm,SymLogNorm);self.assertEqual(norm.vmin,-norm.vmax)
        self.assertEqual(audit['linthresh'],10.);self.assertEqual(norm(0.),.5)
        np.testing.assert_allclose(norm([-100.,-1.])+norm([100.,1.]),1.)

    def test_contrast_guard_fails_the_old_linear_horizon_scale(self):
        image=np.geomspace(1e-6,1.,10000).reshape(100,100);image[0,:]=1e9
        region=np.ones_like(image,dtype=bool)
        fraction,constant=contrast(image,Normalize(0.,np.percentile(image,99.5)),region)
        bad={'lowest_five_percent_fraction':fraction,'constant':constant}
        self.assertGreater(fraction,.9)
        with self.assertRaisesRegex(ValueError,'90%'):check_contrast(bad)
        _,_,good,_=error_colors(image,region);check_contrast(good)
        self.assertLessEqual(good['lowest_five_percent_fraction'],.9)

    def test_zero_missing_and_constant_maps_do_not_invent_error_values(self):
        for image in (np.zeros((3,3)),np.full((3,3),np.nan),np.ones((3,3))):
            before=image.copy();data,norm,audit,notice=error_colors(image,np.ones_like(image,dtype=bool))
            self.assertIsInstance(norm,LogNorm);self.assertGreater(norm.vmin,0);self.assertGreater(norm.vmax,norm.vmin)
            self.assertTrue(notice);check_contrast(audit)
            np.testing.assert_array_equal(image,before)
            if np.all(image==0) or not np.isfinite(image).any():self.assertTrue(np.ma.getmaskarray(data).all())


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
