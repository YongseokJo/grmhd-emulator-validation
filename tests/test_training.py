"""Declared ranges are audited, not accepted as a yes/no claim."""
from pathlib import Path
from unittest.mock import patch

from test_runner import RunnerFixture
from grmhd_validation.artifacts import json_bytes
from grmhd_validation.predictions import read
from grmhd_validation.runner import persistence_request
from grmhd_validation.training import declaration,record_training,training_provenance


class TrainingTests(RunnerFixture):
    def config(self,**changes):
        value={'training_code_commit':'a'*40,'normalization_split':'train','source_split':'synthetic_v1',
               'known_prior_data_exposure':'None declared','prior_data_exposure':[],
               'training_chunks':['train_1','train_2'],'selection_chunks':['validation_1','validation_2']}
        value.update(changes);return {'provenance':value,'seed':42,'epochs':10}

    def test_computes_true_split_training_and_flags_normalization(self):
        computed=declaration(self.config(),self.split)
        self.assertEqual(computed['trained_on_this_split'],'yes')
        self.assertEqual(computed['partition_counts']['training']['train'],6)
        computed=declaration(self.config(normalization_split='all data'),self.split)
        self.assertEqual(computed['trained_on_this_split'],'no');self.assertTrue(computed['normalization_flagged'])

    def test_ranges_union_overlap_and_external_data_recorded_not_rejected(self):
        config=self.config(training_chunks=[],selection_chunks=[],training_dump_ranges=[[0,20],[10,30],[100,104]],selection_dump_ranges=[[31,39]],source_split='historical')
        computed=declaration(config,self.split)
        self.assertEqual(computed['partition_counts']['training'],{'train':3,'validation':14,'test':11,'excluded':3})
        self.assertEqual(computed['outside_split_counts']['training'],5)
        self.assertEqual(computed['label'],'no, overlaps validation/test (25 dumps)')
        self.assertEqual(computed['ranges']['training'],[[0,30],[100,104]])

    def test_unknown_names_and_invalid_ranges_fail(self):
        with self.assertRaisesRegex(ValueError,'Unknown'):declaration(self.config(training_chunks=['mystery']),self.split)
        with self.assertRaisesRegex(ValueError,'inclusive'):declaration(self.config(training_dump_ranges=[[8,7]]),self.split)

    def test_record_reuses_identical_rejects_conflicts_and_binds_split(self):
        checkpoint=self.base/'checkpoint.pt';checkpoint.write_bytes(b'weights')
        request=persistence_request();request.update(model='test-model',checkpoint=str(checkpoint))
        self.request.write_bytes(json_bytes(request))
        config=self.base/'training-config.json';config.write_bytes(json_bytes(self.config()))
        log=self.base/'train.log';log.write_text('Training completed\n')
        with patch('grmhd_validation.slurm.require_compute'):
            result=record_training(self.request,'synthetic_v1',config,log,self.storage(),self.repo)
            self.assertFalse(result['reused'])
            again=record_training(self.request,'synthetic_v1',config,log,self.storage(),self.repo)
            self.assertTrue(again['reused'])
            record=read(result['training_record'])
            self.assertEqual(record['evidence'],'operator declaration');self.assertFalse(record['verified'])
            request['training_record']=result['training_record']
            self.assertEqual(training_provenance(request,self.split)['trained_on_this_split'],'yes')
            config.write_bytes(json_bytes(self.config(normalization_split='all')))
            with self.assertRaisesRegex(FileExistsError,'Conflicting'):record_training(self.request,'synthetic_v1',config,log,self.storage(),self.repo)
            from dataclasses import replace
            with self.assertRaisesRegex(ValueError,'split identity'):training_provenance(request,replace(self.split,sha256='b'*64))
        self.assertEqual(Path(result['training_record']).stat().st_mode&0o777,0o440)
        self.assertEqual(Path(result['training_record']).parent.stat().st_mode&0o7777,0o3770)
