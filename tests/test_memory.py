"""Large-volume replay must keep fields on disk, regardless of start count."""
import gc
import json
from pathlib import Path
import tempfile
import tracemalloc
from types import SimpleNamespace
import unittest

import numpy as np

from grmhd_validation.analyses._analysis_io import FullStore
from grmhd_validation.analyses.common import channel_mean
from grmhd_validation.hashing import sha256


class MemoryTests(unittest.TestCase):
    def test_large_grid_replay_peak_is_independent_of_start_count(self):
        shape=(8,128,64,32)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'state.npy'
            # Constant persistence rollout; use one identical disk artifact for
            # the fixture's frames rather than writing gigabytes of duplicates.
            np.save(path,np.ones(shape,dtype=np.float32))
            ctx=SimpleNamespace(writer=SimpleNamespace(root=root),split=SimpleNamespace(shape=shape))
            store=FullStore(ctx);digest=sha256(path);identity=store.identity(path)
            for start in range(8):
                for step in range(1,33):
                    key=('rollout',start,step)
                    store.paths[key]=path;store.hashes[key]=digest;store.identities[key]=identity
            def peak(count):
                gc.collect();tracemalloc.start();rows=[]
                try:
                    for start in range(count):
                        frames=store.ensure_rollout(start,32)
                        values=[]
                        for step in frames:
                            state=frames[step]
                            self.assertEqual(state.dtype,np.float32)
                            values.append(channel_mean(state.astype(np.float64)))
                        rows.append(np.stack(values))
                        del state,frames,values
                    _,maximum=tracemalloc.get_traced_memory()
                    self.assertTrue(all(row.shape==(32,8) for row in rows))
                    return maximum
                finally:tracemalloc.stop()
            one=peak(1);eight=peak(8);field_bytes=int(np.prod(shape))*4
            print(json.dumps({'one_start_peak_bytes':one,'eight_start_peak_bytes':eight,'field_bytes':field_bytes}),flush=True)
            self.assertLessEqual(eight,one+field_bytes)
            # Also reject a constant-but-large eager 32-frame rollout.
            self.assertLess(max(one,eight),6*field_bytes)
