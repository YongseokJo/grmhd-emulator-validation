"""Real pinned TRON decode regression only; not a benchmark or leaderboard run.

The historical checkpoint overlaps this benchmark's validation. Read the approved
start 4216, pin the fixture SHA256, write results only to caller stdout/temp.
Reference: Kevin Fortune/Sachin, fall26-tron-rc300@2667a5f, main@ebcc218.
"""
import json
from pathlib import Path
import time
import torch

from regression_tron import load_checkpoint
from reference_support import load_reference
from grmhd_validation.adapters.learned import TRON
from grmhd_validation.data import FrameReader,load_geometry
from grmhd_validation.runner import REPO
from grmhd_validation.slurm import require_compute
from grmhd_validation.splits import read_split


def main():
    require_compute();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    fixture=json.loads((REPO/'tests/fixtures/tron_legacy_regression.json').read_text())
    start=time.perf_counter();checkpoint=load_checkpoint(fixture)
    split=read_split(REPO/'splits'/(fixture['split_id']+'.json'))
    reader=FrameReader(split,'validation',Path('/work/nvme/bgun/share/validation/checksums')/(split.split_id+'.sha256'))
    device=torch.device('cuda');geo=load_geometry(split,device,reader.hashes)
    reference,_=load_reference();model=reference['TRON'](in_c=8).to(device).eval()
    model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
    norm=reference['DataNormalizer'](gcov=geo.gcov.float()[None,None],gcon=geo.gcon.float()[None,None]).to(device)
    adapter=TRON(fixture['checkpoint'],geo,str(device),{}).eval()
    history=reader.history(fixture['start'],1,device)
    with torch.inference_mode():
        expected=reference['forecast_one_step'](model,norm,history[:,-1],device)
        actual=adapter.step(history)
    torch.testing.assert_close(actual,expected,rtol=fixture['relative_tolerance'],atol=0)
    relative=((actual.double()-expected.double()).abs()/expected.double().abs().clamp(min=1e-30)).max().item()
    print(json.dumps({'purpose':fixture['purpose'],'checkpoint_sha256':fixture['checkpoint_sha256'],
                      'start':fixture['start'],'status':'pass','max_relative_difference':relative,
                      'seconds':time.perf_counter()-start},sort_keys=True))


if __name__=='__main__':main()
