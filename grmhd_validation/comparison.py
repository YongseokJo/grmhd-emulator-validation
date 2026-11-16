"""One comparison identity for verification and future report/leaderboard rows."""


def comparison_key(info):
    return tuple(info[name] for name in ('split_id','protocol_id','protocol_sha256','tier'))


def report_eligible(path,info):
    """Only sealed, successful runs can enter a report comparison group."""
    return (info.get('status')=='complete' and (path/'COMPLETE').is_file()
            and not (path/'PREVIEW').exists() and not (path/'FAILED').exists()
            and not path.stat().st_mode & 0o222)
