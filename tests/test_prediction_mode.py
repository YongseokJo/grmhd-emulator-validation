"""Host/scorer integrity tests on tiny synthetic grids; no production data writes."""
import copy
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np

from test_runner import RunnerFixture,REPO
from grmhd_validation.artifacts import json_bytes,result_files
from grmhd_validation.hashing import sha256
from grmhd_validation.predictions import (Bundle,task_create,packet_create,packet_advance,packet_collect,read,
                                         required,episode_root,GENERATION_FIELDS)
from grmhd_validation.runner import evaluate
from grmhd_validation.windows import select_starts


class PredictionTests(RunnerFixture):
    def setUp(self):
        super().setUp()
        shutil.copytree(REPO/'prompts',self.repo/'prompts')
        shutil.copytree(REPO/'schemas',self.repo/'schemas')
        self.commit_fixture()
        scheduling=patch('grmhd_validation.slurm.require_compute');scheduling.start();self.addCleanup(scheduling.stop)
        self.task=self.base/'host-task'
        self.bundle=self.base/'bundle'
        self.golden();self.conform()

    def task_make(self,protocol='A',history=1):
        if protocol=='B':
            request=read(self.request);request['options']={'history_length':history}
            self.request.write_bytes(json_bytes(request))
        return task_create(self.request,'synthetic_v1','validation',protocol,
                           'eden-code-v1' if protocol=='A' else 'eden-step-v1', self.task,
                           storage=self.storage(),repo=self.repo)

    def packet(self,kind,start):
        path=self.base/f'packet-{kind}-{start}'
        packet_create(self.task,kind,start,path)
        return path

    def submit(self,packet,delta=0.,source='def step(x): return x\n'):
        episode=read(packet/'episode.json');name=episode['required_prediction_files'][0]
        value=np.load(packet/'inputs/current_state.npy',allow_pickle=False)
        np.save(packet/'submission'/name,value+np.float32(delta),allow_pickle=False)
        code=packet/'submission/code';code.mkdir(exist_ok=True)
        (code/'method.py').write_text(source)
        return packet_advance(self.task,packet)

    def complete(self,protocol='A'):
        self.task_make(protocol)
        task=read(self.task/'task.json')
        for group in task['starts']:
            for kind in ('rollout','single_step'):
                for start in group[kind]:
                    packet=self.packet(kind,start)
                    for step in range(1,(task['max_steps'] if kind=='rollout' else 1)+1):self.submit(packet)
        return packet_collect(self.task,self.bundle)

    def source(self):
        from grmhd_validation.protocol import read_protocol
        config,digest=read_protocol(self.repo/'protocols/forecast_v1.json')
        return Bundle(self.bundle,self.request,self.split,config,digest,'validation',
                      self.shared/'checksums/synthetic_v1.sha256',self.repo)

    def test_task_coverage_and_packet_history_no_future_truth(self):
        self.task_make('B',3)
        task=read(self.task/'task.json');start=task['starts'][0]['rollout'][0]
        packet=self.packet('rollout',start)
        old=np.load(packet/'inputs/history.npy')
        self.assertEqual(old.shape,(3,*self.split.shape))
        self.assertEqual(set(p.name for p in (packet/'inputs').iterdir()),{'history.npy','current_state.npy','geometry.npz'})
        self.assertFalse((packet/'host.json').exists())
        self.assertFalse((packet/'task.json').exists())
        self.assertFalse(any(p.is_symlink() for p in packet.rglob('*')))
        self.submit(packet,10.)
        new=np.load(packet/'inputs/history.npy')
        np.testing.assert_array_equal(new[:-1],old[1:])
        np.testing.assert_array_equal(new[-1],old[-1]+np.float32(10.))
        truth=np.load(self.data/f'dump_{start+1}.npy')
        self.assertFalse(np.array_equal(new[-1],truth))
        self.assertEqual(read(packet/'episode.json')['state_from_step'],1)
        self.assertEqual(len(task['required_targets']),sum(len(g.every)+len(g.rollout) for g in select_starts(self.split,'validation',self.protocol)))

    def test_reject_tampered_inputs_prompt_bad_dtype_and_record_rejections(self):
        self.task_make('B',1);task=read(self.task/'task.json');start=task['starts'][0]['rollout'][0]
        packet=self.packet('rollout',start)
        original=(packet/'prompt.md').read_text();(packet/'prompt.md').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'prompt'):self.submit(packet)
        (packet/'prompt.md').write_text(original)
        name=read(packet/'episode.json')['required_prediction_files'][0]
        np.save(packet/'submission'/name,np.ones(self.split.shape,dtype=np.float64))
        with self.assertRaisesRegex(ValueError,'float32'):packet_advance(self.task,packet)
        self.submit(packet)
        folder=episode_root(self.task,'rollout',task['starts'][0]['chunk'],start)
        record=read(folder/'steps/step_0001.json')
        self.assertEqual(record['validation'],'accepted')
        self.assertEqual(len(list((self.task/'rejections').iterdir())),2)

    def test_partial_coverage_is_not_evaluated_never_published(self):
        self.task_make('B',1)
        result=packet_collect(self.task,self.bundle)
        self.assertFalse(result['complete'])
        source=self.source()
        self.assertFalse(source.availability('rollout_curve')[0])
        self.assertFalse(source.availability('single_step')[0])
        verdict=evaluate(self.request,'synthetic_v1',storage=self.storage(),repo=self.repo,
                         device='cpu',predictions=self.bundle)
        self.assertFalse(verdict['ok']);self.assertEqual(verdict.get('status'),'not evaluated',verdict)
        run=Path(verdict['run_dir'])
        self.assertTrue((run/'NOT_EVALUATED').exists());self.assertFalse((run/'COMPLETE').exists())
        self.assertFalse((self.shared/'runs/synthetic_v1').exists())
        summary=read(run/'summary.json')
        self.assertEqual(summary['rollout_curve']['status'],'not evaluated')
        self.assertEqual(read(run/'run.json')['predictions']['attestation'],'unattested')

    def test_complete_import_uses_common_metrics_identical_to_persistence(self):
        self.complete('A');source=self.source()
        self.assertTrue(source.availability('single_step')[0]);self.assertTrue(source.availability('rollout_curve')[0])
        self.assertFalse(source.attested)
        adapter=evaluate(self.request,'synthetic_v1',storage=self.storage(),repo=self.repo,device='cpu')
        imported=evaluate(self.request,'synthetic_v1',storage=self.storage(),repo=self.repo,device='cpu',predictions=self.bundle)
        self.assertTrue(adapter['ok'],adapter);self.assertTrue(imported['ok'],imported)
        self.assertEqual(adapter['result_fingerprint'],imported['result_fingerprint'])
        self.assertEqual(result_files(Path(adapter['run_dir'])),result_files(Path(imported['run_dir'])))

    def test_rollout_step_one_does_not_replace_single_step_coverage(self):
        self.task_make('B',1);task=read(self.task/'task.json')
        for group in task['starts']:
            for start in group['rollout']:
                packet=self.packet('rollout',start)
                for _ in range(task['max_steps']):self.submit(packet)
        packet_collect(self.task,self.bundle);source=self.source()
        self.assertTrue(source.availability('rollout_curve')[0]);self.assertFalse(source.availability('single_step')[0])
        result=evaluate(self.request,'synthetic_v1',storage=self.storage(),repo=self.repo,device='cpu',predictions=self.bundle)
        self.assertEqual(result.get('status'),'not evaluated',result)
        run=Path(result['run_dir']);self.assertTrue((run/'figures/fig02_rollout_curve.png').exists())
        self.assertEqual(read(run/'summary.json')['rollout_curve']['n_starts'],4)

    def test_modified_frozen_prediction_is_invalid_not_silently_replaced(self):
        self.complete('B');manifest=read(self.bundle/'predictions.json');path=self.bundle/manifest['entries'][0]['path']
        path.chmod(0o640);np.save(path,np.ones(self.split.shape,dtype=np.float32))
        source=self.source();self.assertTrue(source.invalid)
        self.assertFalse(source.availability(manifest['entries'][0]['kind'] if manifest['entries'][0]['kind']=='single_step' else 'rollout_curve')[0])

    def test_symlink_and_reused_session_rejected(self):
        self.complete('B');(self.bundle/'extra-link').symlink_to(self.data)
        with self.assertRaisesRegex(ValueError,'symlink'):self.source()

    def test_real_split_requires_exactly_600_rollout_entries(self):
        from grmhd_validation.splits import read_split
        from grmhd_validation.protocol import read_protocol
        protocol,_=read_protocol(REPO/'protocols/forecast_v1.json')
        for split_id in ('rc300_reduced_v1','5000run_v1'):
            split=read_split(REPO/'splits'/(split_id+'.json'))
            for partition in ('validation','test'):
                starts=select_starts(split,partition,protocol)
                self.assertEqual(sum(len(g.rollout) for g in starts),12)
                self.assertEqual(sum(len(g.rollout)*protocol['max_steps'] for g in starts),600)

    def test_resume_after_acceptance_does_not_replace_accepted_output(self):
        self.task_make('B',1);task=read(self.task/'task.json');start=task['starts'][0]['rollout'][0]
        packet=self.packet('rollout',start)
        with patch('grmhd_validation.predictions.render_packet',side_effect=RuntimeError('simulated crash after freeze')):
            with self.assertRaisesRegex(RuntimeError,'simulated'):self.submit(packet,1.)
        folder=episode_root(self.task,'rollout',task['starts'][0]['chunk'],start)
        before=sha256(folder/'prediction_step_0001.npy')
        packet_create(self.task,'rollout',start,packet)
        self.assertEqual(read(packet/'episode.json')['active_step'],2)
        self.assertEqual(sha256(folder/'prediction_step_0001.npy'),before)
        self.submit(packet,2.)
        self.assertEqual(sha256(folder/'prediction_step_0001.npy'),before)

    def test_b_tools_can_change_per_step_with_recorded_hashes(self):
        self.task_make('B',1);task=read(self.task/'task.json');start=task['starts'][0]['rollout'][0]
        packet=self.packet('rollout',start)
        self.submit(packet,source='method one\n');self.submit(packet,source='method two\n')
        folder=episode_root(self.task,'rollout',task['starts'][0]['chunk'],start)
        first=read(folder/'steps/step_0001.json');second=read(folder/'steps/step_0002.json')
        self.assertNotEqual(first['code_sha256'],second['code_sha256'])
        self.assertNotEqual(first['code_directory'],second['code_directory'])
        result=packet_collect(self.task,self.bundle)
        self.assertFalse(result['coverage']['invalid'])

    def test_reused_b_session_invalidates_whole_rollout_coverage(self):
        self.task_make('B',1);task=read(self.task/'task.json');start=task['starts'][0]['rollout'][0]
        packet=self.packet('rollout',start)
        folder=episode_root(self.task,'rollout',task['starts'][0]['chunk'],start)
        for _ in range(2):
            generation=read(folder/'generation.json');generation['session_id']='reused-session'
            (folder/'generation.json').write_bytes(json_bytes(generation));self.submit(packet)
        packet_collect(self.task,self.bundle);source=self.source()
        self.assertTrue(any('Session was reused' in row['reason'] for row in source.invalid))
        self.assertFalse(source.availability('rollout_curve')[0])

    def test_hash_updated_metadata_still_cannot_forge_input_chain(self):
        self.complete('B');manifest=read(self.bundle/'predictions.json');entry=manifest['entries'][0]
        path=self.bundle/entry['record_path'];record=read(path);record['input_history_sha256']='0'*64
        path.chmod(0o640);path.write_bytes(json_bytes(record));entry['record_sha256']=sha256(path)
        mpath=self.bundle/'predictions.json';mpath.chmod(0o640);mpath.write_bytes(json_bytes(manifest))
        source=self.source();self.assertTrue(any('chain mismatch' in row['reason'] for row in source.invalid))

    def test_host_attestation_is_bound_to_manifest_and_missing_evidence_stays_visible(self):
        self.complete('B')
        from grmhd_validation.predictions import ATTESTATIONS
        path=self.bundle/'host_attestation.json';attestation=read(path)
        self.assertEqual(attestation['predictions_sha256'],sha256(self.bundle/'predictions.json'))
        self.assertFalse(self.source().attested)
        attestation.update(operator='operator',implementation_commit='a'*40,attested_at='2026-11-17T00:00:00Z',
                           isolation_mechanism='isolated workspace',evidence_sha256=['b'*64],claims={k:True for k in ATTESTATIONS})
        path.chmod(0o640);path.write_bytes(json_bytes(attestation))
        # Generation provider/session evidence is still unknown: host booleans
        # alone never transform an anonymous packet into a claimed blind result.
        self.assertFalse(self.source().attested)

    def test_atomic_frozen_record_never_publishes_partial_json(self):
        from grmhd_validation.predictions import write
        path=self.base/'immutable.json'
        with patch('grmhd_validation.predictions.os.link',side_effect=OSError('simulated publication failure')):
            with self.assertRaisesRegex(OSError,'publication'):write(path,{'value':1},True)
        self.assertFalse(path.exists());self.assertFalse(list(self.base.glob('.pending-*')))
        write(path,{'value':1},True)
        with self.assertRaises(FileExistsError):write(path,{'value':2},True)
        self.assertEqual(read(path),{'value':1})

    def test_agent_cannot_change_inputs_then_rewrite_episode_hashes(self):
        self.task_make('B',1);task=read(self.task/'task.json');start=task['starts'][0]['rollout'][0]
        packet=self.packet('rollout',start);episode=read(packet/'episode.json')
        path=packet/'inputs/current_state.npy';value=np.load(path);np.save(path,value+1)
        episode['input_hashes']['current_state.npy']=sha256(path)
        (packet/'episode.json').write_bytes(json_bytes(episode))
        with self.assertRaisesRegex(ValueError,'frozen host input'):self.submit(packet)
