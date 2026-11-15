"""Required Slurm-only regression; NOT a validation/leaderboard submission.

The pinned checkpoint saw this benchmark's validation data during its older
80/10/10 training split. It is used exclusively to verify the numerical port.
All outputs live in TemporaryDirectory; no shared result namespace is opened.
Source: Kevin Fortune/Sachin, main@ebcc218 and fall26-tron-rc300@2667a5f.
"""
import argparse
import contextlib
import importlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch
from matplotlib.figure import Figure
from unittest.mock import patch

from reference_support import load_reference,observe_return
from grmhd_validation.analyses._analysis_io import FullStore,geometry_views
from grmhd_validation.analyses._port_manifest import ANALYSES
from grmhd_validation.adapter import ModelAdapter
from grmhd_validation.artifacts import Writer,json_bytes
from grmhd_validation.data import FrameReader,load_geometry
from grmhd_validation.hashing import sha256
from grmhd_validation.permissions import SharedStorage,GROUP_AREAS,OWNER_AREAS
from grmhd_validation.protocol import read_protocol
from grmhd_validation.runner import Context,initialize
from grmhd_validation.slurm import require_compute
from grmhd_validation.splits import read_split
from grmhd_validation.windows import select_starts

REPO=Path(__file__).resolve().parents[1]

# Public result fields -> independently observed ORIGINAL evaluator locals.
# No expected value is computed with port functions or parsed from print output.
FIELDS={
 'error_correlation':('evaluate_error_correlation',{'model':'corr_model','persistence':'corr_data','model_errors':'err_model','persistence_errors':'err_data'}),
 'stability':('evaluate_stability',{'draws':'ls','mean':'L_mean','std':'L_std','skipped_draws':'skipped_draws'}),
 'cost':('evaluate_cost',{'parameters':'n_params'}),
 'physics_table':('evaluate_physics_table',{'initial_divb':'d0','initial_normalized_divb':'nd0','initial_normalized_divb_global':'nd0_glob'}),
 'shell_profiles':('evaluate_shell_profiles',{'r':'r_mid'}),
 'temporal_consistency':('evaluate_temporal_consistency',{'steps':'steps','mass_pred':'M_p','mass_truth':'M_t','energy_pred':'E_p','energy_truth':'E_t','angular_pred':'L_p','angular_truth':'L_t','mass_scale':'scale_M','energy_scale':'scale_E','angular_scale':'scale_L'}),
 'conservation_closure':('evaluate_conservation_closure',{'pred_bare':'R_b','truth_bare':'T_b','pred_physical':'R_ph','truth_physical':'T_ph','imbalance_pred_bare':'imb_p_b','imbalance_truth_bare':'imb_t_b','imbalance_pred_physical':'imb_p_ph','imbalance_truth_physical':'imb_t_ph'}),
 'accretion_diagnostics':('evaluate_accretion_diagnostics',{'flux_profiles':'saved_phi','jet_power_profiles':'saved_pjet','phi_bh':'saved_phi_bh','mdot_pred':'mdot_p','mdot_truth':'mdot_t','steps':'steps','horizon_index':'r_H_idx','r10_index':'r10'}),
 'mdot':('plot_mdot',{'r':'r_mid'}),
 'lorentz':('evaluate_lorentz',{'max':'gmax','max_step':'gmax_step','max_p999':'p999_max','clamp_count':'n_clamp'}),
 'mean_profiles':('plot_mean_profiles',{'r':'r3','theta':'h3','omega_pred':'omega_p','omega_truth':'omega_t','omega_keplerian':'omega_k','rho_theta_pred':'rho_p_norm','rho_theta_truth':'rho_t_norm','beta_log10_pred':'beta_p','beta_log10_truth':'beta_t','sigma_log10_pred':'sigma_p','sigma_log10_truth':'sigma_t','n_steps':'n_steps'}),
 'statistical_props':('evaluate_statistical_props',{'channels':'metrics','step':'n_steps'}),
 'spectrum':('plot_spectrum',{'rho_modes':'k0t','rho_ratio':'ratio_rho','rho_ratio_mask':'m_rho','B1_modes':'kt2','B1_ratio':'ratio_br','B1_ratio_mask':'m_br','step':'n_steps'}),
 'azimuthal_spectra':('plot_azimuthal_spectra',{'modes':'m_axis','truth':'truth_specs','pred':'pred_specs','steps':'horizons'}),
 'coherence':('plot_coherence',{'steps':'tt','coherence':'coh'}),
 'temporal_spectrum':('plot_temporal_power_spectrum',{'frequencies':'freqs','pred_disk':'psd_pred_disk','truth_disk':'psd_true_disk','pred_jet':'psd_pred_jet','truth_jet':'psd_true_jet','n_steps':'T_avail'}),
 'turbulence_stats':('plot_turbulence_stats',{'step':'n_steps'}),
 'conditional_error':('evaluate_conditional_error',{'edges':'edges','model':'err_m','persistence':'err_p','counts':'cnt'}),
 'bias_analysis':('evaluate_bias_analysis',{'bias':'bias','rmse':'rmse','n_samples':'n'}),
 'spatial_error':('evaluate_spatial_error',{}),
 'meridional_slice':('plot_meridional_slice',{'X':'X','Y':'Y','rho_error':'log_rho_e','B2_pred_log10':'b2_p','B2_truth_log10':'b2_t','B2_error':'b2_e','step':'target_step'}),
 'rollout_evolution':('plot_rollout_evolution',{'X':'X','Y':'Y'}),
}
LAST_SAMPLE={
 'physics_table':{'step':'h','mass_error':'eM','energy_error':'eE','angular_error':'eL','divb_pred':'dp','divb_truth':'dt','normalized_divb_pred':'ndp','normalized_divb_truth':'ndt','flux_pred':'flux_p','flux_truth':'flux_t','gamma_max':'gmax'},
 'shell_profiles':{'step':'n_steps','mass_pred':'M_p','mass_truth':'M_t','energy_pred':'E_p','energy_truth':'E_t','angular_pred':'L_p','angular_truth':'L_t','mass_error':'sl_err_M','energy_error':'sl_err_E','angular_error':'sl_err_L'},
 'mdot':{'step':'h','pred':'M_p','truth':'M_t','relative_error':'err'},
 'lorentz':{'step':'step','gamma_max':'curr_gmax'},
 'statistical_props':{'channel':'c','centers':'centers','pdf_pred':'pp','pdf_truth':'pt','widths':'widths','js':'js'},
 'spectrum':{'channel':'ch','modes':'kt','power_pred':'sp','power_truth':'st','slope_pred':'sl_p','slope_truth':'sl_t','fit_r2_pred':'r2_p','fit_r2_truth':'r2_t'},
 'turbulence_stats':{'channel':'c','exponents_pred':'zp','exponents_truth':'zt','fit_r2_pred':'r2p','fit_r2_truth':'r2t','s3_pred':'s3p','s3_truth':'s3t'},
 'spatial_error':{'step':'n_steps','radius':'centers','disk':'prof_disk','jet':'prof_jet'},
 'rollout_evolution':{'step':'hh','truth':'Z_t','pred':'Z_f','error':'Z_err'},
}


def check_checkpoint(fixture):
    path=Path(fixture['checkpoint'])
    if not path.is_file():raise FileNotFoundError('Required regression checkpoint missing: '+str(path))
    actual=sha256(path)
    if actual!=fixture['checkpoint_sha256']:
        raise ValueError('Required regression checkpoint SHA256 mismatch: '+actual)


def load_checkpoint(fixture):
    """Hash and deserialize the same open file; reject concurrent replacement."""
    path=Path(fixture['checkpoint'])
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno())
        digest=hashlib.sha256()
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=fixture['checkpoint_sha256']:raise ValueError('Required regression checkpoint SHA256 mismatch')
        stream.seek(0)
        result=torch.load(stream,map_location='cpu',weights_only=True)
        after=os.fstat(stream.fileno());current=path.stat()
        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if identity(before)!=identity(after) or identity(after)!=identity(current):
            raise ValueError('Regression checkpoint changed while loading')
    return result


def compare(actual,expected,label,tolerance,counts):
    if isinstance(expected,torch.Tensor):expected=expected.detach().cpu().numpy()
    if isinstance(actual,torch.Tensor):actual=actual.detach().cpu().numpy()
    if isinstance(expected,dict):
        expected={str(k):v for k,v in expected.items()};actual={str(k):v for k,v in actual.items()}
        if set(actual)!=set(expected):raise AssertionError(label+' keys differ')
        for key in expected:compare(actual[key],expected[key],label+'.'+key,tolerance,counts)
    elif isinstance(expected,(list,tuple)):
        if len(actual)!=len(expected):raise AssertionError(label+' lengths differ')
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,label+'.'+str(i),tolerance,counts)
    else:
        np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=0,equal_nan=True,err_msg=label)
        counts[0]+=np.asarray(expected).size


def run(fixture,kind,base,device):
    start=fixture['start'];steps=fixture['max_steps']
    split=read_split(REPO/'splits'/(fixture['split_id']+'.json'))
    protocol,_=read_protocol(REPO/'protocols/forecast_v1.json')
    initialize(protocol,device)
    # Exactly 51 source-evaluator dumps, plus geometry. History remains accessible
    # only through FrameReader, not through the original evaluator's directory.
    links=base/'original-inputs';links.mkdir()
    dumps={d.index:d for d in split.model_dumps(fixture['partition'])}
    for index in range(start,start+steps+1):
        (links/('dump_'+str(index)+'.npy')).symlink_to(split.dump_path(dumps[index]))
    (links/'geometry.npz').symlink_to(split.geometry_path)
    assert len(list(links.iterdir()))==52
    storage_root=base/'artifacts';storage_root.mkdir();storage_root.chmod(0o2750)
    for name in OWNER_AREAS+GROUP_AREAS:
        (storage_root/name).mkdir();(storage_root/name).chmod(0o2750 if name in OWNER_AREAS else 0o3770)
    import grp
    storage=SharedStorage(storage_root,grp.getgrgid(os.getegid()).gr_name)
    output=storage.folder('staging','regression-only')
    reader=FrameReader(split,fixture['partition'],Path('/work/nvme/bgun/share/validation/checksums')/(split.split_id+'.sha256'))
    geometry=load_geometry(split,torch.device('cpu'),reader.hashes)
    ev,source_tree=load_reference()
    geo=ev['load_geometry'](str(links/'geometry.npz'),torch.device('cpu'))
    ev['CONFIG'].update(eos_gamma=geo['eos_gamma'],dt_dump=geo['dt_dump'])
    original_norm=ev['DataNormalizer'](gcov=geo['gcov'],gcon=geo['gcon'])
    gpu_norm=ev['DataNormalizer'](gcov=geo['gcov'].to(device),gcon=geo['gcon'].to(device)).to(device)
    criterion=ev['PhysicsLoss3D'](torch.device('cpu'),str(links/'geometry.npz'))
    original_step=ev['forecast_one_step']
    if kind=='tron':
        model=ev['TRON'](in_c=8).to(device)
        checkpoint=load_checkpoint(fixture)
        model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
    else:
        class Zero(torch.nn.Module):
            def forward(self,value):return torch.zeros_like(value)
        model=Zero().to(device)
    model.eval()
    class RegressionAdapter(ModelAdapter):
        name='legacy-regression-only';version='v1';history_length=10;max_batch=1;requires_checkpoint=True
        def __init__(self):self.model=model
        def step(self,history):
            if kind=='persistence':return history[:,-1].clone()
            return original_step(model,gpu_norm,history[:,-1].to(device),device)
        def lorentz_clamp_count(self,reset=True):
            return 0 if kind=='persistence' else gpu_norm.lorentz_clamp_count(reset=reset)
    adapter=RegressionAdapter()
    # The host executes decode on GPU while the original metric normalizer stays
    # on CPU. Its clamp diagnostic must observe that same decoder, too.
    original_norm.lorentz_clamp_count=adapter.lorentz_clamp_count
    ctx=Context(adapter,geometry,reader,select_starts(split,fixture['partition'],protocol),
                protocol,split,fixture['partition'],device,Writer(storage,output))
    ctx.regression_ids=list(range(start,start+steps+1));ctx.run_info={};ctx.full=FullStore(ctx)
    assert ctx.starts[0].rollout[0]==start
    # Both evaluators call the same model-owned physical decode. The reference
    # math runs on CPU; its original functions are otherwise unchanged.
    def forecast(model_arg,norm_arg,current,cpu):
        history=reader.history(start,adapter.history_length,device)
        history[:,-1]=current.to(device)
        return ctx.step(history).cpu()
    ev['forecast_one_step']=forecast
    ev['CONFIG']['geometry_file']=str(links/'geometry.npz')
    original_out=base/'original-results';original_out.mkdir()
    args=SimpleNamespace(start=start,max_steps=steps,n_starts=1,seed=protocol['seed'],save_dir=str(original_out))
    report={};failures=[]
    for name in ANALYSES:
        begin=time.perf_counter();count=[0]
        print(kind,name,flush=True)
        try:
            function,fields=FIELDS[name]
            adapter.lorentz_clamp_count(reset=True)
            with (base/(name+'-original.log')).open('w') as log,contextlib.redirect_stdout(log), \
                 patch.object(Figure,'savefig',return_value=None):
                _,locals_=observe_return(ev[function],args,model,original_norm,original_norm,
                                        criterion,geo,np.load(links/'geometry.npz'),torch.device('cpu'),str(links),
                                        loop_fields=LAST_SAMPLE.get(name),tree=source_tree)
            with patch('grmhd_validation.analyses._analysis_io.save_figure',return_value=None):
                module=importlib.import_module('grmhd_validation.analyses.'+name)
                adapter.lorentz_clamp_count(reset=True)
                # Modules hold their imported save function directly.
                with patch.object(module,'save_figure',return_value=None):
                    actual=module.compute(ctx,space='loss',starts=None)['per_start'][0]['values']
            for field,variable in fields.items():
                compare(actual[field],locals_[variable],name+'.'+field,fixture['relative_tolerance'],count)
            if name in LAST_SAMPLE:
                expected_rows=locals_['_observed_iterations']
                if len(expected_rows)!=len(actual['samples']):raise AssertionError(name+' source iteration count differs')
                for index,(a,b) in enumerate(zip(actual['samples'],expected_rows)):
                    compare({key:a[key] for key in b},b,name+'.samples.'+str(index),fixture['relative_tolerance'],count)
            report[name]={'status':'pass','numbers_compared':count[0],'seconds':time.perf_counter()-begin}
        except Exception as error:
            import traceback
            traceback.print_exc()
            failures.append(name);report[name]={'status':'fail','error':str(error),'seconds':time.perf_counter()-begin}
        print(json.dumps({name:report[name]}),flush=True)
    # Prediction horizon/tau-c have separately parameterized legacy sampling.
    from grmhd_validation.analyses.correlation_times import compute,decorrelation
    count=[0];begin=time.perf_counter()
    try:
        with (base/'correlation-original.log').open('w') as log,contextlib.redirect_stdout(log):
            _,expected=observe_return(ev['evaluate_correlation_times'],args,model,original_norm,original_norm,
                                     criterion,geo,np.load(links/'geometry.npz'),torch.device('cpu'),str(links))
        actual=compute(ctx,space='loss',starts=[start],legacy=True)['per_start'][0]['values']
        for field,variable in {'tau_c':'tau_c','saturated':'sat','crossing_steps':'crossing','never_crossed':'never_crossed',
                               'last_positive_steps':'last_pos','fraction_positive_after_first':'frac_pos_after',
                               'mean_skill_25_50':'mean_s25_50','skill':'skill','steps':'valid_steps'}.items():
            compare(actual[field],expected[variable],'correlation_times.'+field,fixture['relative_tolerance'],count)
        report['correlation_times']={'status':'pass','numbers_compared':count[0],'seconds':time.perf_counter()-begin}
    except Exception as error:
        import traceback
        traceback.print_exc();failures.append('correlation_times')
        report['correlation_times']={'status':'fail','error':str(error)}
    ctx.full.cleanup()
    return {'kind':kind,'status':'fail' if failures else 'pass','failures':failures,'analyses':report}


def main():
    require_compute()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,default=REPO/'tests/fixtures/tron_legacy_regression.json')
    parser.add_argument('--kind',choices=['persistence','tron','both'],default='both')
    args=parser.parse_args();fixture=json.loads(args.fixture.read_text());check_checkpoint(fixture)
    if os.environ.get('SLURM_JOB_PARTITION')!='ghx4-interactive':raise RuntimeError('Regression requires ghx4-interactive')
    reports=[]
    with tempfile.TemporaryDirectory(prefix='gval-regression-only-') as temporary:
        for kind in (['persistence','tron'] if args.kind=='both' else [args.kind]):
            base=Path(temporary)/kind;base.mkdir()
            reports.append(run(fixture,kind,base,torch.device('cuda')))
        print(json.dumps({'purpose':fixture['purpose'],'checkpoint_sha256':fixture['checkpoint_sha256'],
                          'reports':reports},sort_keys=True),flush=True)
    if any(report['status']!='pass' for report in reports):raise SystemExit(1)


if __name__=='__main__':main()
