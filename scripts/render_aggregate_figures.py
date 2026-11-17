"""Regenerate presentation from verified stored metrics, without model inference."""
import os
os.umask(0o007)
import argparse
import gc
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import numpy as np
import torch

from grmhd_validation.artifacts import Writer,fingerprint,result_files
from grmhd_validation.data import load_geometry
from grmhd_validation.hashing import sha256
from grmhd_validation.permissions import SharedStorage
from grmhd_validation.protocol import read_protocol
from grmhd_validation.runner import REPO,git_commit
from grmhd_validation.slurm import require_compute
from grmhd_validation.splits import read_split,CHANNELS
from grmhd_validation.analyses._port_manifest import FIGURES
from grmhd_validation.analyses._figures import render,validate_figure


def main():
    require_compute()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--receipt-dir',type=Path,help='Copy contact sheet and fig18 into a repository docs directory')
    args=parser.parse_args()
    source=args.source.resolve();info=json.loads((source/'run.json').read_text())
    files=result_files(source);before=fingerprint(files)
    if before!=info['result_fingerprint']:raise ValueError('Source result fingerprint mismatch')
    protocol,digest=read_protocol(REPO/'protocols/forecast_v1.json')
    if digest!=info['protocol_sha256']:raise ValueError('Source protocol differs')
    split=read_split(REPO/'splits'/(info['split_id']+'.json'))
    if split.sha256!=info['split_file_sha256']:raise ValueError('Source frozen split differs')
    storage=SharedStorage();storage.require_owner()
    root=storage.exclusive_folder('reports',split.split_id,'figure-review-'+uuid.uuid4().hex[:12])
    writer=Writer(storage,root)
    ctx=SimpleNamespace(split=split,partition=info.get('split','validation'),protocol=protocol,
                        constants=protocol['constants'],writer=writer,channels=CHANNELS,
                        geometry=load_geometry(split,torch.device('cpu')))
    import matplotlib
    matplotlib.use(ctx.constants['figures']['backend'])
    import matplotlib.pyplot as plt
    audit={}
    for name,outputs in FIGURES.items():
        if not outputs:continue
        metric=json.loads((source/'metrics'/(name+'.json')).read_text())
        audit.update(render(ctx,name,metric['data']['aggregate']))
        del metric
        gc.collect()
    # The quick-tier accuracy figure is also part of every full-tier bundle.
    # Its per-start error curves are retained by correlation_times, so rendering
    # requires neither re-inference nor a scan of ground-truth dumps.
    from grmhd_validation.analyses import rollout_curve
    metric=json.loads((source/'metrics/rollout_curve.json').read_text())
    correlation=json.loads((source/'metrics/correlation_times.json').read_text())
    rows=correlation['data']['per_start']
    models=np.asarray([row['values']['nmae_model'] for row in rows],dtype=np.float64)
    persistence=np.asarray([row['values']['nmae_persistence'] for row in rows],dtype=np.float64)
    audit['fig02_rollout_curve.png']=rollout_curve.plot(ctx,np.asarray(metric['mean_nmae']),np.asarray(metric['std_nmae']),
                          np.asarray(metric['mean_persist']),np.asarray(metric['per_channel']),models,persistence)
    del metric,correlation,rows,models,persistence
    # A contact sheet is a review artifact, not a replacement for full-size PNGs.
    names=sorted(audit);columns=3;rows=(len(names)+columns-1)//columns
    figure,axes=plt.subplots(rows,columns,figsize=(18,rows*4),squeeze=False)
    try:
        for ax,name in zip(axes.flat,names):
            pixels=plt.imread(root/'figures'/name)
            stride=max(1,int(max(pixels.shape[1]/850,pixels.shape[0]/550)))
            ax.imshow(pixels[::stride,::stride]);ax.set_title(name,fontsize=10);ax.axis('off')
            del pixels
        for ax in list(axes.flat)[len(names):]:ax.axis('off')
        figure.suptitle(f'{split.split_id} / {ctx.partition} — all {len(names)} full-tier figures',fontsize=16)
        figure.tight_layout(rect=(0,0,1,.98))
        with writer.open('contact_sheet.png') as stream:figure.savefig(stream,dpi=150,metadata={'Software':None})
    finally:plt.close(figure)
    if fingerprint(result_files(source))!=before:raise RuntimeError('Source result content changed during rendering')
    writer.json('figure_labels.json',audit)
    receipt={'ok':True,'purpose':'figure presentation review','source':str(source),'output_dir':str(root),
             'source_result_fingerprint':before,'protocol_sha256':digest,'renderer_repo_commit':git_commit(REPO),
             'renderer_sha256':{name:sha256(REPO/'grmhd_validation/analyses'/name) for name in ('_figures.py','rollout_curve.py')},
             'figures':{name:sha256(root/'figures'/name) for name in names},'contact_sheet_sha256':sha256(root/'contact_sheet.png')}
    writer.json('presentation.json',receipt)
    if args.receipt_dir:
        destination=args.receipt_dir.resolve()
        if not destination.is_relative_to(REPO/'docs'):raise ValueError('Receipt copies must be under repository docs/')
        destination.mkdir(parents=True,exist_ok=True)
        for src,dst in ((root/'contact_sheet.png',split.split_id+'_contact_sheet.png'),
                        (root/'figures/fig18_accretion_diagnostics.png',split.split_id+'_fig18.png')):
            target=destination/dst
            if target.is_symlink():raise ValueError('Receipt target must not be a symlink')
            shutil.copyfile(src,target)
    print(json.dumps(receipt,sort_keys=True))


if __name__=='__main__':main()
