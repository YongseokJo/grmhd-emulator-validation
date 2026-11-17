"""Presentation acceptance for every full-tier PNG, including negative oracles."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import importlib

import numpy as np

from test_full_analyses import FullFixture
from grmhd_validation.analyses._figures import validate_figure,step_label,STEP_COLORS
from grmhd_validation.analyses._port_manifest import FIGURES


class FigurePresentationTests(FullFixture):
    def test_every_full_tier_figure_has_physics_labels_and_bounded_bands(self):
        ctx=self.context();seen={}
        def inspect(fig,*args,**kwargs):
            audit=validate_figure(fig);seen[fig.get_label()]=audit
            self.assertTrue(audit['panels'])
            self.assertTrue(any(ctx.split.split_id in text and ctx.partition in text and '95%' in text for text in audit['caption']))
            for ax in fig.axes:
                for collection in ax.collections:
                    if (collection.get_gid() or '').startswith('ci95:'):
                        self.assertLessEqual(collection.get_alpha(),.15)
                        self.assertTrue(np.all(np.asarray(collection.get_linewidths())==0))
            if fig.get_label()=='fig18_accretion_diagnostics.png':
                self.assertEqual([panel['ci_bands'] for panel in audit['panels']],[2,2,2])
                self.assertEqual([panel['mean_curves'] for panel in audit['panels']],[2*len(ctx.protocol['report_steps'])]*2+[2])
                for ax in fig.axes[:2]:
                    for artist in ax.lines:
                        if not (artist.get_gid() or '').startswith('mean:'):continue
                        step=int(artist.get_label().split('step ')[1].split()[0])
                        self.assertIn(step,ctx.protocol['report_steps'])
                        self.assertEqual(artist.get_color(),STEP_COLORS.get(step,('#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#17becf')[step%8]))
                        self.assertIn(step_label(ctx,step),artist.get_label())
                        self.assertEqual(artist.get_linestyle(),'-' if artist.get_label().startswith('Truth') else '--')
                        self.assertNotIn('step 0 ',artist.get_label())
                self.assertEqual(audit['panels'][0]['xscale'],'log');self.assertEqual(audit['panels'][0]['yscale'],'log')
                self.assertEqual(audit['panels'][1]['yscale'],'symlog')
                self.assertIn('T^r',audit['panels'][1]['ylabel'])
                self.assertIn('95% CI at step 8 only',audit['caption'][0])
            if fig.get_label()=='fig05_shell_profiles.png':self.assertEqual([p['ci_bands'] for p in audit['panels']],[2,1]*3)
            if fig.get_label()=='fig07_mdot.png':self.assertEqual([p['ci_bands'] for p in audit['panels']],[2,1])
            if fig.get_label()=='fig12_azimuthal_spectra.png':self.assertEqual([p['ci_bands'] for p in audit['panels']],[2]*3+[1]*3)
            if fig.get_label()=='fig02_rollout_curve.png':self.assertEqual(audit['panels'][0]['ci_bands'],2)
        with patch('matplotlib.figure.Figure.savefig',autospec=True,side_effect=inspect):
            for name in ctx.protocol['tiers']['full']:
                importlib.import_module('grmhd_validation.analyses.'+name).run(ctx)
        expected={filename for files in FIGURES.values() for filename in files.values()}|{'fig02_rollout_curve.png'}
        self.assertEqual(set(seen),expected)
        ctx.full.cleanup()


class FigureLabelOracles(unittest.TestCase):
    def test_undefined_structure_curves_keep_log_scale_without_inventing_values(self):
        import matplotlib.pyplot as plt
        from grmhd_validation.analyses._figures import panel
        fig,ax=plt.subplots()
        try:
            ax.plot([1,2],[np.nan,np.nan],label='Truth')
            panel(ax,'Radial Structure Function','Radial lag [cells]','Normalized structure function',xscale='log',yscale='log')
            fig.tight_layout()
            self.assertEqual(ax.get_yscale(),'log')
            self.assertTrue(np.isnan(ax.lines[0].get_ydata()).all())
            self.assertTrue(any('No finite positive values' in text.get_text() for text in ax.texts))
        finally:plt.close(fig)

    def test_raw_titles_raw_legends_and_unlabeled_axes_fail(self):
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        try:
            ax.set(title='Magnetic Flux Threading',xlabel=r'$r\;[R_g]$',ylabel=r'$\Phi$ [code units]')
            validate_figure(fig)
            ax.set_title('profiles_pred magnetic_flux 0')
            with self.assertRaisesRegex(ValueError,'Placeholder'):validate_figure(fig)
            ax.set_title('Magnetic Flux Threading');ax.plot([1,2],[1,2],label='profiles_truth magnetic_flux 0');ax.legend()
            with self.assertRaisesRegex(ValueError,'Placeholder'):validate_figure(fig)
            ax.get_legend().remove();ax.set_xlabel('')
            with self.assertRaisesRegex(ValueError,'Unlabeled'):validate_figure(fig)
        finally:plt.close(fig)

    def test_step_label_uses_dump_time_not_array_position(self):
        text=step_label(SimpleNamespace(split=SimpleNamespace(dt_dump=100.)),10)
        self.assertEqual(text,r'step 10 (1,000 $R_g/c$)')
