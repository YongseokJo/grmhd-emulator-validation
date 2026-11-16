"""Oracles separating physical averaging from display and Fourier averaging."""
from unittest.mock import patch
import numpy as np
from test_full_analyses import FullFixture


class PhysicalProfileTests(FullFixture):
    def test_mean_profiles_keep_physical_values_before_display_logs(self):
        from grmhd_validation.analyses._profiles import mean_profiles
        ctx=self.context();start=ctx.starts[0].rollout[0]
        frames={step:np.zeros(ctx.split.shape,dtype=np.float32) for step in range(3)}
        for step,frame in frames.items():frame[0]=[1.,4.,9.][step];frame[1]=[1.,10.,100.][step];frame[5]=1.
        with patch.object(ctx.reader,'numpy',side_effect=lambda index:frames[index-start]),patch.object(ctx.full,'ensure_rollout',return_value=frames):
            values=mean_profiles(ctx,start)
        for side in ('pred','truth'):
            np.testing.assert_allclose(values['rho_theta_'+side],14./3.,rtol=1e-13)
            np.testing.assert_allclose(values['beta_'+side],2*(5./3.-1)*37.,rtol=1e-13)
            np.testing.assert_allclose(values['sigma_'+side],3./14.,rtol=1e-13)

    def test_temporal_power_averages_cell_power_not_region_mean_power(self):
        from grmhd_validation.analyses._profiles import temporal_spectrum
        ctx=self.context();start=ctx.starts[0].rollout[0];count=ctx.protocol['max_steps']
        initial=np.zeros(ctx.split.shape,dtype=np.float32);initial[0]=2.;initial[1]=.1;initial[5]=1.
        signs=np.where(np.indices(ctx.split.shape[1:])[-1]%2,1.,-1.)
        frames={step:initial.copy() for step in range(1,count+1)}
        for step,value in frames.items():value[0]+=signs*np.sin(2*np.pi*step/count)
        with patch.object(ctx.reader,'numpy',side_effect=lambda index:frames[index-start]),patch.object(ctx.full,'ensure_rollout',return_value=frames):
            values=temporal_spectrum(ctx,start)
        signal=np.asarray([np.sin(2*np.pi*step/count) for step in range(1,count+1)])
        window=np.hanning(count);expected=np.abs(np.fft.rfft((signal-signal.mean())*window))**2/((1/25.)*np.sum(window**2))
        np.testing.assert_allclose(values['truth_disk'][0],expected,rtol=2e-6,atol=1e-12)
        self.assertGreater(values['truth_disk'][0,1],0.)
        np.testing.assert_array_equal(values['pred_disk'],values['truth_disk'])

    def test_truth_histogram_edges_are_independent_of_model_predictions(self):
        from grmhd_validation.analyses._reference_statistics import histogram_edges
        ctx=self.context();channel=0;step=ctx.protocol['max_steps']//ctx.constants['full']['distribution_step_divisor']
        values=np.concatenate([ctx.reader.numpy(s+step)[channel].astype(np.float64).ravel() for g in ctx.starts for s in g.rollout])
        low,high=np.percentile(np.maximum(np.abs(values),ctx.constants['full']['epsilon']),ctx.constants['full']['histogram_log_percentiles'])
        with patch.object(ctx.adapter,'step',side_effect=AssertionError('Truth bins cannot call a model')):
            edges=histogram_edges(ctx,channel)
        np.testing.assert_allclose(edges,np.geomspace(low,high,ctx.constants['full']['histogram_bins']+1),rtol=1e-14)
