"""Quick content stays invariant when full semantics acquire a new hash cohort."""
from grmhd_validation.artifacts import json_bytes
from grmhd_validation.conformance import reference
from test_runner import RunnerFixture
from unittest.mock import patch
from pathlib import Path
import json
from grmhd_validation.artifacts import result_files
from grmhd_validation.hashing import sha256
from grmhd_validation.conformance import refresh_reference_figures


class ReferenceCohortTests(RunnerFixture):
    def test_identical_result_reference_refreshes_pngs_without_changing_metrics(self):
        first=self.golden();root=Path(first['reference_dir']);image=root/'figures/fig02_rollout_curve.png'
        before=result_files(root);old_image=sha256(image);old_mode=image.stat().st_mode
        from matplotlib.figure import Figure
        save=Figure.savefig
        def revised(figure,*args,**kwargs):
            figure.text(.5,.5,'Updated presentation',ha='center')
            return save(figure,*args,**kwargs)
        with patch('matplotlib.figure.Figure.savefig',autospec=True,side_effect=revised):second=self.golden()
        self.assertEqual(first['result_fingerprint'],second['result_fingerprint'])
        self.assertEqual(before,result_files(root));self.assertNotEqual(old_image,sha256(image))
        self.assertEqual(image.stat().st_mode,old_mode)
        record=json.loads((root/'figures/presentation.json').read_text())
        self.assertEqual(record['figures'][image.name],sha256(image))
        with self.assertRaisesRegex(ValueError,'fingerprint differs'):
            refresh_reference_figures(self.storage(),root,root/'figures','0'*64)
        self.assertEqual(before,result_files(root))

    def test_presentation_refresh_rejects_completed_model_run_paths(self):
        with self.assertRaisesRegex(ValueError,'owner persistence reference'):
            refresh_reference_figures(self.storage(),self.shared/'runs/synthetic/model/completed',self.base,'0'*64)

    def test_identical_content_hashes_pass_without_numeric_reparsing(self):
        from grmhd_validation.conformance import compare_results
        roots=[self.base/'first-result',self.base/'second-result']
        for root in roots:
            root.mkdir();(root/'summary.json').write_bytes(b'{"value": 1.25}\n')
        with patch('grmhd_validation.conformance.numeric_difference',side_effect=AssertionError('Identical hashes need no numerical traversal')):
            self.assertEqual(compare_results(*roots),{'ok':True,'differing_files':[],'largest_relative_difference':0.})

    def test_final_quick_reference_cannot_accept_changed_legacy_content(self):
        path=self.repo/'protocols/forecast_v1.json'
        self.protocol['constants']['full']['production_ready']=False
        path.write_bytes(json_bytes(self.protocol));legacy=self.golden()
        self.protocol['constants']['full']['production_ready']=True
        path.write_bytes(json_bytes(self.protocol));final=self.golden()
        self.assertEqual(legacy['result_fingerprint'],final['result_fingerprint'])
        self.assertNotEqual(legacy['reference_dir'],final['reference_dir'])
        self.protocol['constants']['loss_space']['floor_rho']=1000.
        path.write_bytes(json_bytes(self.protocol))
        with self.assertRaisesRegex(ValueError,'Quick result bytes changed'):
            reference('synthetic_v1',storage=self.storage(),repo=self.repo)
        self.assertEqual(list((self.shared/'references').rglob('.candidate-*')),[])
