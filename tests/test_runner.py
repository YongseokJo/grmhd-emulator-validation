"""Synthetic, compute-node tests of the full milestone-2 contract."""
import copy
import getpass
import grp
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from grmhd_validation.adapter import adapter_source, validate_declaration, validate_output
from grmhd_validation.artifacts import fingerprint, json_bytes, result_files
from grmhd_validation.checksums import checksum_bytes
from grmhd_validation.conformance import conformance, reference, reference_path
from grmhd_validation.data import FrameReader, load_geometry
from grmhd_validation.environment import platform_id
from grmhd_validation.hashing import sha256
from grmhd_validation.permissions import GROUP_AREAS, OWNER_AREAS, SharedStorage
from grmhd_validation.protocol import read_protocol
from grmhd_validation.runner import evaluate, persistence_request
from grmhd_validation.splits import CHANNELS, SPLIT_NAMES, read_split
from grmhd_validation.verify import compare_runs, seal_run, verify_run
from grmhd_validation.windows import select_starts

REPO = Path(__file__).resolve().parents[1]


class RunnerFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.addCleanup(self.cleanup)
        self.repo, self.data, self.shared = (self.base / name for name in ('repo', 'data', 'shared'))
        self.data.mkdir()
        for name in ('splits', 'protocols', 'environment'):
            (self.repo / name).mkdir(parents=True)
        (self.repo / 'environment/platforms.json').write_bytes((REPO / 'environment/platforms.json').read_bytes())
        self.protocol = json.loads((REPO / 'protocols/forecast_v1.json').read_text())
        self.protocol.update(H_max=3, max_steps=4, report_steps=[1, 2, 4], n_starts_per_chunk=2,
                             bootstrap_block=3, bootstrap_resamples=30)
        (self.repo / 'protocols/forecast_v1.json').write_bytes(json_bytes(self.protocol))
        shape = (2, 2, 2)
        coordinates = np.indices(shape, dtype=np.float64)
        metric = np.broadcast_to(np.diag([-1., 1., 1., 1.])[:, :, None, None, None], (4, 4, *shape)).copy()
        np.savez(self.data / 'geometry.npz', r=coordinates[0] + 8, h=coordinates[1] + 1,
                 ph=coordinates[2], gcov=metric, gcon=metric, gdet=np.ones(shape),
                 resolved=np.ones(shape, bool), dx=np.ones(3), spin=np.array(.94), dt_dump=np.array(25.), eos_gamma=np.array(5./3.))
        self.spec = dict(schema_version=2, split_id='synthetic_v1', dataset='synthetic', data_root=str(self.data),
                         geometry_file='geometry.npz', geometry_sha256=sha256(self.data / 'geometry.npz'), times_file=None,
                         array=dict(shape=[8, *shape], dtype='float32', channels=list(CHANNELS), axes=['channel','r','theta','phi']),
                         dt_dump=25., spin=.94, total_dumps=43, segments=[],
                         splits={s: dict(count=0, chunks=[], files=[]) for s in SPLIT_NAMES})
        for name, first, last in [('train',0,2),('excluded',3,3),('validation',4,14),('excluded',15,15),
                                  ('test',16,26),('excluded',27,27),('validation',28,39),('train',40,42)]:
            section = self.spec['splits'][name]
            chunk = dict(chunk=f'{name}_{len(section["chunks"])+1}', first=first, last=last, count=last-first+1)
            self.spec['segments'].append(dict(chunk, split=name))
            section['chunks'].append(chunk)
            section['count'] += chunk['count']
            for index in range(first, last+1):
                filename = f'dump_{index}.npy'
                section['files'].append(filename)
                # Every physical value encodes its dump; excluded frames are poisonous.
                frame = np.arange(64, dtype=np.float32).reshape(8,*shape)/64 + 1 + index/8
                if name == 'excluded':
                    frame.fill(np.nan)
                np.save(self.data / filename, frame)
        self.file = self.repo / 'splits/synthetic_v1.json'
        self.file.write_bytes(json_bytes(self.spec))
        self.shared.mkdir()
        self.shared.chmod(0o2750)
        for name in OWNER_AREAS + GROUP_AREAS:
            (self.shared / name).mkdir()
            (self.shared / name).chmod(0o2750 if name in OWNER_AREAS else 0o3770)
        (self.shared / 'splits/synthetic_v1.json').write_bytes(self.file.read_bytes())
        self.group = grp.getgrgid(os.getegid()).gr_name
        self.split = read_split(self.file)
        (self.shared / 'checksums/synthetic_v1.sha256').write_bytes(checksum_bytes(self.split, 1))
        self.request = self.base / 'request.json'
        self.request.write_bytes(json_bytes(persistence_request()))
        self.old_umask = os.umask(0o007)
        self.addCleanup(os.umask, self.old_umask)
        # Scheduling itself is separately enforced/tested; all these tests run in Slurm.
        self.compute = patch('grmhd_validation.runner.require_compute')
        self.compute.start(); self.addCleanup(self.compute.stop)
        self.compute2 = patch('grmhd_validation.conformance.require_compute')
        self.compute2.start(); self.addCleanup(self.compute2.stop)
        self.commit_fixture()

    def commit_fixture(self):
        import subprocess
        subprocess.run(['git','init','-q',str(self.repo)],check=True)
        subprocess.run(['git','-C',str(self.repo),'add','.'],check=True)
        subprocess.run(['git','-C',str(self.repo),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','Synthetic fixture','--allow-empty'],check=True)

    def cleanup(self):
        for parent, dirs, _ in os.walk(self.base):
            Path(parent).chmod(0o700)
            for name in dirs:
                (Path(parent)/name).chmod(0o700)
        self.temp.cleanup()

    def storage(self):
        return SharedStorage(self.shared, self.group)

    def golden(self):
        return reference('synthetic_v1', storage=self.storage(), repo=self.repo)

    def conform(self):
        return conformance('synthetic_v1', storage=self.storage(), repo=self.repo)

    def run_eval(self, **kwargs):
        return evaluate(self.request, 'synthetic_v1', storage=self.storage(), repo=self.repo, device='cpu', **kwargs)

    def adapter(self, body, name='custom', history=3):
        path = self.base / 'adapter.py'
        path.write_text('from grmhd_validation import ModelAdapter\nimport torch\nclass Adapter(ModelAdapter):\n'
                        f' name="{name}"\n version="v1"\n history_length={history}\n max_batch=1\n requires_checkpoint=False\n'
                        ' def __init__(self, checkpoint, geometry, device, options):\n  self.geometry=geometry\n'
                        ' def step(self, history):\n' + body + '\n')
        request = persistence_request()
        request.update(model=name, adapter=str(path)+':Adapter')
        self.request.write_bytes(json_bytes(request))
        return path


class RunnerTests(RunnerFixture):
    def test_run_ids_are_split_specific_and_attempt_scan_is_one_level(self):
        from grmhd_validation.runstore import reserve
        storage=self.storage()
        args=(storage,'persistence','v1',None,'synthetic_v1','validation','quick')
        first,name,number=reserve(*args)
        self.assertEqual(name,'persistence-v1__nockpt__synthetic_v1__validation__quick__a1')
        self.assertEqual(number,1)
        # Decoys inside attempts and conformance must never be visited/counted.
        (first/name.replace('__a1','__a99')).mkdir()
        conformance_dir=storage.folder('staging',getpass.getuser(),'conformance')
        (conformance_dir/name.replace('__a1','__a98')).mkdir()
        other_model=storage.destination('synthetic_v1','other-model')
        (other_model/name.replace('__a1','__a97')).mkdir()
        second,_,number=reserve(*args)
        self.assertEqual(number,2)
        destination=storage.destination('synthetic_v1','persistence')
        second.rename(destination/second.name)
        peer=storage.folder('staging','another-user')
        (peer/name.replace('__a1','__a3')).mkdir()
        _,other,number=reserve(storage,'persistence','v1',None,'different_v1','validation','quick')
        self.assertEqual(number,1);self.assertIn('__different_v1__',other)
        original=Path.iterdir
        def shallow(path):
            self.assertNotIn(path,{first,conformance_dir,other_model,destination/second.name})
            return original(path)
        with patch('grmhd_validation.runstore.os.walk',side_effect=AssertionError('No recursive attempt scan')), \
                patch.object(Path,'iterdir',shallow):
            _,_,number=reserve(*args)
        self.assertEqual(number,4)

    def test_common_starts_excluded_and_chunk_boundaries(self):
        starts = select_starts(self.split, 'validation', self.protocol)
        self.assertEqual(list(starts[0].every), list(range(6,11)))
        self.assertEqual(starts[0].rollout, (6,10))
        self.assertEqual(starts[1].rollout, (30,35))
        reader = FrameReader(self.split, 'validation', self.shared/'checksums/synthetic_v1.sha256')
        for excluded in (3,15,27,16,40):
            with self.assertRaisesRegex(ValueError, 'outside'):
                reader.numpy(excluded)
        with self.assertRaisesRegex(ValueError, 'boundary'):
            reader.history(4,3,'cpu')
        shortened = copy.deepcopy(self.protocol); shortened['max_steps'] = 10
        with self.assertRaises(ValueError):
            select_starts(self.split,'validation',shortened)

    def test_golden_conformance_repeat_evaluate_and_verify(self):
        first = self.golden()
        self.assertTrue(first['ok']); self.assertTrue(first['provisional'])
        second = self.golden()
        self.assertEqual(first['files'], second['files'])
        result = self.conform()
        self.assertTrue(result['ok'], result)
        self.assertFalse(list((self.shared/'runs').iterdir()))
        self.assertEqual(result['result_fingerprint'], first['result_fingerprint'])
        a, b = self.run_eval(), self.run_eval()
        self.assertTrue(a['ok'], a); self.assertTrue(b['ok'], b)
        self.assertTrue(a['run_id'].endswith('__a1')); self.assertTrue(b['run_id'].endswith('__a2'))
        left, right = Path(a['run_dir']), Path(b['run_dir'])
        self.assertEqual(result_files(left), result_files(right))
        self.assertEqual(sha256(left/'figures/fig02_rollout_curve.png'), sha256(right/'figures/fig02_rollout_curve.png'))
        import struct
        png=(left/'figures/fig02_rollout_curve.png').read_bytes()
        self.assertEqual(struct.unpack('>II',png[16:24]),(2100,750))
        self.assertNotIn(b'tIME',png)
        self.assertTrue(verify_run(left,self.repo)['ok'])
        self.assertTrue(compare_runs(left,right,self.repo)['ok'])
        run = json.loads((left/'run.json').read_text())
        self.assertIsNone(run['checkpoint_sha256']); self.assertIn('__nockpt__',run['run_id'])
        self.assertEqual(run['conformance']['result_fingerprint'], result['result_fingerprint'])
        self.assertTrue(run['data_verified'])
        for path in [left,*left.rglob('*')]:
            self.assertEqual(stat.S_IMODE(path.stat().st_mode)&0o222,0)
        before = (self.shared/'runs/synthetic_v1/persistence').stat().st_mode
        self.assertEqual(stat.S_IMODE(before),0o3770)
        checks = json.loads((left/'checksums.json').read_text())
        checks2 = json.loads((right/'checksums.json').read_text())
        self.assertEqual({k:v for k,v in checks.items() if k in first['files']}, {k:v for k,v in checks2.items() if k in first['files']})
        summary = json.loads((left/'summary.json').read_text())
        self.assertEqual(summary['single_step']['n_samples'],11)
        self.assertEqual(summary['rollout_curve']['n_starts'],4)
        self.assertEqual(summary['single_step']['mean']['skill'],0.)
        self.assertEqual(summary['rollout_curve']['at_steps']['4']['skill'],0.)
        for name in first['files']:
            content=(left/name).read_text()
            for forbidden in ('run_id','start_time','end_time','host','version','attempt'):
                self.assertNotIn('"'+forbidden+'"',content)

    def test_missing_golden_and_required_conformance_record_failed_attempt(self):
        result = self.conform()
        self.assertFalse(result['ok']); self.assertEqual(result['next'],'ask the owner to run scripts/reference_delta.sbatch')
        self.assertTrue((Path(result['output_dir'])/'FAILED').is_file())
        evaluation = self.run_eval()
        self.assertFalse(evaluation['ok']); self.assertTrue((Path(evaluation['run_dir'])/'FAILED').is_file())
        self.assertFalse(list((self.shared/'runs').iterdir()))
        self.golden(); self.conform()
        success = self.run_eval()
        self.assertTrue(success['ok'],success); self.assertTrue(success['run_id'].endswith('__a2'))

    def test_autoregression_history_and_excluded_never_reach_adapter(self):
        self.golden(); self.conform()
        self.adapter('  assert history.shape[1]==3\n  assert torch.isfinite(history).all()\n  assert not hasattr(self.geometry,"data_root")\n  assert torch.allclose(history[:,1:]-history[:,:-1],torch.full_like(history[:,1:],0.125))\n  assert not torch.is_grad_enabled()\n  return history[:,-1]+0.125')
        result=self.run_eval()
        self.assertTrue(result['ok'],result)
        summary=json.loads((Path(result['run_dir'])/'summary.json').read_text())
        self.assertEqual(summary['single_step']['mean']['nmae_model'],0.)
        self.assertEqual(summary['rollout_curve']['at_steps']['4']['mean_nmae'],0.)
        run=json.loads((Path(result['run_dir'])/'run.json').read_text())
        golden_path=reference_path(self.storage(),'synthetic_v1',self.protocol,platform_id())
        baseline=json.loads((golden_path/'run.json').read_text())
        self.assertEqual(run['rollout_starts'],baseline['rollout_starts'])

    def test_bad_output_checkpoint_and_failed_attempts_stay_in_staging(self):
        self.golden(); self.conform()
        for body, message in [('  return history[:,-1].double()','dtype'),
                              ('  return history[:,-1,:,:1]','shape'),
                              ('  return history[:,-1]*float("nan")','non-finite')]:
            self.adapter(body)
            verdict=self.run_eval()
            self.assertFalse(verdict['ok']); self.assertIn(message,verdict['error'])
            output=Path(verdict['run_dir'])
            self.assertTrue((output/'FAILED').exists()); self.assertFalse((output/'COMPLETE').exists())
            self.assertTrue((output/'logs/evaluate.log').read_text())
        self.adapter('  return history[:,-1]')
        verdict=self.run_eval(); self.assertTrue(verdict['ok'],verdict)
        self.assertTrue(verdict['run_id'].endswith('__a4'))
        request=persistence_request(); request['checkpoint']=str(self.data/'dump_0.npy')
        self.request.write_bytes(json_bytes(request))
        verdict=self.run_eval(); self.assertFalse(verdict['ok']); self.assertIn('requires_checkpoint',verdict['error'])

    def test_hash_missing_mismatch_and_explicit_unverified(self):
        self.golden(); self.conform()
        manifest=self.shared/'checksums/synthetic_v1.sha256'
        original=manifest.read_text()
        manifest.write_text('\n'.join(line for line in original.splitlines() if not line.endswith('dump_6.npy'))+'\n')
        missing=self.run_eval(); self.assertFalse(missing['ok']); self.assertIn('missing',missing['error'].lower())
        manifest.write_text(original)
        frame=self.data/'dump_6.npy'; raw=np.load(frame); np.save(frame,raw+0.125)
        corrupt=self.run_eval(); self.assertFalse(corrupt['ok']); self.assertIn('mismatch',corrupt['error'].lower())
        unverified=self.run_eval(no_verify_data=True)
        self.assertTrue(unverified['ok'],unverified); self.assertFalse(unverified['data_verified'])
        self.assertTrue(json.loads((Path(unverified['run_dir'])/'run.json').read_text())['no_verify_data'])

    def test_golden_refuses_changed_results_and_cross_platform_tolerance(self):
        result=self.golden()
        target=Path(result['reference_dir'])
        hashes=result_files(target)
        raw=np.load(self.data/'dump_6.npy'); np.save(self.data/'dump_6.npy',raw+0.25)
        (self.shared/'checksums/synthetic_v1.sha256').write_bytes(checksum_bytes(self.split,1))
        with self.assertRaisesRegex(ValueError,'Golden differs'):
            self.golden()
        self.assertEqual(result_files(target),hashes)
        np.save(self.data/'dump_6.npy',raw)
        (self.shared/'checksums/synthetic_v1.sha256').write_bytes(checksum_bytes(self.split,1))
        other='delta-x86_64' if platform_id()!='delta-x86_64' else 'deltaai-aarch64'
        target.rename(target.with_name(other))
        verdict=self.conform()
        self.assertTrue(verdict['ok'],verdict); self.assertEqual(verdict['status'],'warn'); self.assertTrue(verdict['cross_platform'])

    def test_atomic_publish_collision_preserves_failure_and_existing_modes(self):
        self.golden(); self.conform()
        staging_mode=stat.S_IMODE((self.shared/'staging').stat().st_mode)
        with patch('grmhd_validation.runstore.no_replace_rename',side_effect=OSError('rename failed')):
            result=self.run_eval()
        self.assertFalse(result['ok']); root=Path(result['run_dir'])
        self.assertTrue((root/'FAILED').is_file()); self.assertFalse((root/'COMPLETE').exists())
        self.assertEqual(stat.S_IMODE((self.shared/'staging').stat().st_mode),staging_mode)
        retry=self.run_eval(); self.assertTrue(retry['ok'],retry); self.assertTrue(retry['run_id'].endswith('__a2'))

    def unsealed_run(self):
        self.golden(); self.conform()
        with patch.object(SharedStorage, 'seal_created_root', side_effect=OSError('interrupted after rename')):
            result=self.run_eval()
        self.assertFalse(result['ok']); self.assertTrue(result['unsealed'])
        root=Path(result['run_dir'])
        self.assertIn('/runs/',str(root)); self.assertTrue(root.is_dir())
        self.assertEqual(result['next'],'gval seal '+str(root))
        self.assertFalse((self.shared/'staging'/getpass.getuser()/root.name).exists())
        self.assertTrue((root/'COMPLETE').is_file()); self.assertFalse((root/'FAILED').exists())
        self.assertTrue(root.stat().st_mode & stat.S_IWUSR)
        self.assertTrue(all(not p.stat().st_mode & 0o222 for p in root.rglob('*')))
        return root

    def test_unsealed_run_is_excluded_until_owner_recovers_with_cli_seal(self):
        import contextlib
        import io
        from grmhd_validation.cli import main
        root=self.unsealed_run()
        verdict=verify_run(root,self.repo)
        self.assertFalse(verdict['ok']); self.assertTrue(verdict['unsealed'])
        with patch('grmhd_validation.verify.verify_run',side_effect=lambda path:verify_run(path,self.repo)), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['verify',str(root)]),1)
        self.assertTrue(json.loads(output.getvalue().splitlines()[-1])['unsealed'])
        with self.assertRaisesRegex(ValueError,'unsealed'):
            compare_runs(root,root,self.repo)
        before={p.relative_to(root).as_posix():(p.stat().st_ino,p.stat().st_mode,p.stat().st_mtime_ns,
                sha256(p) if p.is_file() else None) for p in root.rglob('*')}
        root_inode=root.stat().st_ino
        with patch('grmhd_validation.verify.seal_run',side_effect=lambda path:seal_run(path,self.repo)), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['seal',str(root)]),0)
        self.assertTrue(json.loads(output.getvalue().splitlines()[-1])['ok'])
        self.assertEqual(root.stat().st_ino,root_inode)
        self.assertEqual(root.stat().st_mode & 0o222,0)
        after={p.relative_to(root).as_posix():(p.stat().st_ino,p.stat().st_mode,p.stat().st_mtime_ns,
               sha256(p) if p.is_file() else None) for p in root.rglob('*')}
        self.assertEqual(before,after)
        self.assertTrue(verify_run(root,self.repo)['ok'])
        with patch('grmhd_validation.verify.os.fchmod',side_effect=AssertionError('Already sealed')):
            self.assertTrue(seal_run(root,self.repo)['ok'])

    def test_seal_refuses_foreign_owner_mutable_contents_bad_hash_and_missing_complete(self):
        root=self.unsealed_run()
        with patch('grmhd_validation.verify.os.geteuid',return_value=os.geteuid()+1), \
                patch('grmhd_validation.verify.os.fchmod',side_effect=AssertionError('Foreign chmod')):
            with self.assertRaisesRegex(PermissionError,'owner'):
                seal_run(root,self.repo)
        metric=root/'metrics/single_step.json'
        metric.chmod(0o640)
        with self.assertRaisesRegex(ValueError,'not read-only'):
            seal_run(root,self.repo)
        metric.chmod(0o440)
        metric.parent.chmod(0o3770)
        with self.assertRaisesRegex(ValueError,'not read-only'):
            seal_run(root,self.repo)
        metric.parent.chmod(0o3550)
        original=metric.read_bytes();metric.chmod(0o640);metric.write_bytes(original+b' ');metric.chmod(0o440)
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            seal_run(root,self.repo)
        metric.chmod(0o640);metric.write_bytes(original);metric.chmod(0o440)
        marker=(root/'COMPLETE').read_bytes();(root/'COMPLETE').unlink()
        with self.assertRaisesRegex(ValueError,'COMPLETE'):
            seal_run(root,self.repo)
        (root/'COMPLETE').write_bytes(marker);(root/'COMPLETE').chmod(0o440)
        link=root.with_name('symlink-run');link.symlink_to(root,target_is_directory=True)
        with self.assertRaises(OSError):
            seal_run(link,self.repo)
        link.unlink()
        special=root/'unexpected-fifo';os.mkfifo(special,mode=0o440)
        with self.assertRaisesRegex(ValueError,'special file'):
            seal_run(root,self.repo)
        special.unlink()
        self.assertTrue(root.stat().st_mode & stat.S_IWUSR)
        self.assertTrue(seal_run(root,self.repo)['ok'])

    def test_publication_freezes_contents_before_rename_and_seals_root_afterward(self):
        from grmhd_validation.runstore import no_replace_rename
        self.golden();self.conform()
        def rename(source,target):
            self.assertTrue(source.stat().st_mode & stat.S_IWUSR)
            self.assertTrue((source/'COMPLETE').is_file())
            self.assertTrue(all(not p.stat().st_mode & 0o222 for p in source.rglob('*')))
            no_replace_rename(source,target)
            self.assertTrue(target.stat().st_mode & stat.S_IWUSR)
        with patch('grmhd_validation.runstore.no_replace_rename',side_effect=rename):
            result=self.run_eval()
        self.assertTrue(result['ok'],result)
        self.assertEqual(Path(result['run_dir']).stat().st_mode & 0o222,0)

    def test_geometry_hash_and_adapter_declarations(self):
        geometry=load_geometry(self.split,'cpu')
        self.assertEqual(geometry.shape,(2,2,2)); self.assertEqual(geometry.resolved.dtype,torch.bool)
        cls,source=adapter_source('persistence')
        validate_declaration(cls,self.protocol)
        self.assertEqual(len(source['source_sha256']),64)
        cls=type('TooLong',(cls,),{'history_length':4})
        with self.assertRaisesRegex(ValueError,'history_length'):
            validate_declaration(cls,self.protocol)
        (self.data/'geometry.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'frozen split'):
            load_geometry(self.split,'cpu')

    def test_same_platform_conformance_requires_exact_bytes(self):
        result=self.golden(); path=Path(result['reference_dir'])
        metric=path/'metrics/single_step.json'
        payload=json.loads(metric.read_text()); payload['per_channel']['rho']['nmae_model']*=1+1e-13
        metric.write_bytes(json_bytes(payload))
        summary=json.loads((path/'summary.json').read_text());summary['metric_sha256']['single_step']=sha256(metric)
        (path/'summary.json').write_bytes(json_bytes(summary))
        record=json.loads((path/'golden.json').read_text()); record['files']=result_files(path); record['result_fingerprint']=fingerprint(record['files'])
        (path/'golden.json').write_bytes(json_bytes(record))
        verdict=self.conform()
        self.assertFalse(verdict['ok']); self.assertIn('metrics/single_step.json',verdict['differing_files'])
        self.assertGreater(verdict['largest_relative_difference'],0.)


if __name__=='__main__':
    unittest.main()
