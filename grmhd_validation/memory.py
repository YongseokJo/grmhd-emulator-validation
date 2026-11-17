"""Operational Linux RSS measurements, kept outside deterministic results."""
from contextlib import contextmanager
from pathlib import Path
import os
import resource
import threading
import time


def discard_file_cache(path):
    """Release clean sequential-I/O pages charged to the Slurm memory cgroup."""
    if hasattr(os,'posix_fadvise'):
        with Path(path).open('rb') as stream:
            os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)


def cgroup_memory():
    """Include file cache: Slurm accounts more than process anonymous RSS."""
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            root=Path('/sys/fs/cgroup')/line[3:].lstrip('/')
            # Slurm places the process below the step cgroup.
            while root!=Path('/sys/fs/cgroup'):
                if root.name.startswith('step_') and (root/'memory.current').is_file():
                    stats=dict(line.split() for line in (root/'memory.stat').read_text().splitlines())
                    return {'total':int((root/'memory.current').read_text()),'anon':int(stats['anon']),'file':int(stats['file'])}
                root=root.parent
    return {}


def rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    raise RuntimeError('Linux VmRSS is unavailable')


@contextmanager
def measure(records,name):
    initial=rss_bytes();peak=[initial];charged=cgroup_memory();stop=threading.Event();begin=time.monotonic()
    def poll():
        while not stop.wait(.02):
            peak[0]=max(peak[0],rss_bytes())
            for key,value in cgroup_memory().items():charged[key]=max(charged.get(key,0),value)
    worker=threading.Thread(target=poll,daemon=True);worker.start()
    try:yield
    finally:
        stop.set();worker.join();final=rss_bytes()
        records[name]={'peak_rss_bytes':max(peak[0],final),'initial_rss_bytes':initial,
                       'final_rss_bytes':final,'process_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       'cgroup_peak_bytes':charged,'seconds':time.monotonic()-begin}
