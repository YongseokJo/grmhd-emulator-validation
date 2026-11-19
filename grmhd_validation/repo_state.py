"""Fail-closed inspection of the benchmark's protected Git paths."""
from pathlib import Path
import subprocess

PROTECTED = ('splits/', 'protocols/', 'grmhd_validation/', 'schemas/', 'prompts/')


class RepositoryModifiedError(RuntimeError):
    def __init__(self, info):
        self.state = {key: info[key] for key in ('repo_modified', 'repo_status')}
        super().__init__('Protected repository paths are modified or cannot be inspected; commit reviewed changes before evaluation')


def inspect(repo):
    try:
        return _inspect(repo)
    except OSError as error:
        return {'repo_modified': True, 'repo_status': ['ERROR: Git inspection failed: '+str(error)]}


def _inspect(repo):
    repo = Path(repo).resolve()
    root = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
    if root.returncode or Path(root.stdout.strip()).resolve() != repo:
        return {'repo_modified': True, 'repo_status': ['ERROR: evaluation requires a Git repository root']}
    result = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain=v1', '--untracked-files=all', '--', *PROTECTED], capture_output=True, text=True)
    if result.returncode:
        return {'repo_modified': True, 'repo_status': ['ERROR: git status failed: '+result.stderr.strip()]}
    entries = result.stdout.splitlines()
    return {'repo_modified': bool(entries), 'repo_status': entries}


def require_clean(info, repo):
    current = inspect(repo)
    info['repo_modified'] = bool(info.get('repo_modified')) or current['repo_modified']
    info['repo_status'] = sorted(set(info.get('repo_status', [])) | set(current['repo_status']))
    if info['repo_modified']:
        raise RepositoryModifiedError(info)
