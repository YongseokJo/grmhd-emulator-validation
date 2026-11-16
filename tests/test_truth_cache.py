"""Owner build/read conformance and corruption guards for shared truth caches."""
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from test_full_analyses import FullFixture
from grmhd_validation.conformance import reference,conformance
from grmhd_validation.hashing import sha256
from grmhd_validation.protocol import read_protocol
from grmhd_validation.truth_cache import TruthCache,cache_path,identity,encode,decode


class TruthCacheTests(FullFixture):
    def test_build_flushes_each_entry_before_the_next_calculation(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=TruthCache(directory,building=True)
            try:
                expected=np.arange(100,dtype=np.float32)
                np.testing.assert_array_equal(cache.value('incremental',{},lambda:expected),expected)
                self.assertFalse(cache.connection.in_transaction)
                with sqlite3.connect(Path(directory)/'quantities.sqlite3') as independent:
                    self.assertEqual(independent.execute('SELECT count(*) FROM quantities').fetchone()[0],1)
            finally:cache.close()
    def test_full_reference_cache_is_read_only_and_reproduces_result_bytes(self):
        result=reference('synthetic_v1',tier='full',storage=self.storage(),repo=self.repo)
        self.assertTrue(result['ok'],result)
        protocol,digest=read_protocol(self.repo/'protocols/forecast_v1.json')
        path=cache_path(self.storage(),self.split,'validation',protocol,digest)
        expected=identity(self.split,'validation',protocol,digest,self.shared/'checksums/synthetic_v1.sha256')
        before={p.name:sha256(p) for p in path.iterdir()}
        cache=TruthCache(path,expected=expected)
        with self.assertRaisesRegex(ValueError,'missing'):cache.value('invented',{},lambda:1.)
        cache.close()
        verdict=conformance('synthetic_v1',storage=self.storage(),repo=self.repo,tier='full')
        self.assertTrue(verdict['ok'],verdict)
        self.assertEqual(result['result_fingerprint'],verdict['result_fingerprint'])
        self.assertEqual(before,{p.name:sha256(p) for p in path.iterdir()})
        info=json.loads((Path(verdict['output_dir'])/'run.json').read_text())
        self.assertEqual(info['truth_cache_sha256'],json.loads((path/'manifest.json').read_text())['cache_sha256'])
        with self.assertRaisesRegex(ValueError,'identity'):
            TruthCache(path,expected={**expected,'protocol_sha256':'0'*64})
        with (path/'quantities.sqlite3').open('ab') as stream:stream.write(b'corruption')
        with self.assertRaisesRegex(ValueError,'SHA256'):TruthCache(path,expected=expected)

    def test_failed_owner_build_removes_only_its_own_candidates(self):
        root=self.shared/'references/previous-candidate';root.mkdir();(root/'keep').write_text('owner file')
        with patch('grmhd_validation.runner.execute',side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                reference('synthetic_v1',tier='full',storage=self.storage(),repo=self.repo)
        self.assertTrue((root/'keep').exists())
        self.assertEqual(list((self.shared/'references').rglob('.candidate-*')),[])

    def test_pickle_free_round_trip_preserves_keys_dtypes_and_nonfinite_values(self):
        original={1:np.array([np.nan,2.],dtype=np.float64),'reason':np.array(['','nonpositive_mass_inflow']),
                  'tuple':(True,None,np.array([1,2],dtype=np.int64))}
        actual=decode(encode(original));self.assertEqual(set(original),set(actual))
        np.testing.assert_array_equal(actual[1],original[1]);self.assertEqual(actual[1].dtype,np.float64)
        np.testing.assert_array_equal(actual['reason'],original['reason']);self.assertIsInstance(actual['tuple'],tuple)
        with self.assertRaises(ValueError):encode(np.array([object()],dtype=object))
