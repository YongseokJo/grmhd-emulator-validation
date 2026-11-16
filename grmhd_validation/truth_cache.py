"""Owner-built, immutable-by-hash truth/persistence quantities; no pickle."""
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
import torch

from .artifacts import json_bytes
from .hashing import sha256
from .validation import schema_validate


def encode(value):
    arrays={}
    def pack(item):
        if isinstance(item,dict):return {'dict':[[pack(k),pack(v)] for k,v in sorted(item.items(),key=lambda kv:str(kv[0]))]}
        if isinstance(item,(tuple,list)):return {'tuple' if isinstance(item,tuple) else 'list':[pack(v) for v in item]}
        if item is None:return {'none':True}
        tensor=isinstance(item,torch.Tensor)
        array=item.detach().cpu().numpy() if tensor else np.asarray(item)
        if array.dtype.hasobject:raise ValueError('Object arrays are forbidden in truth caches')
        key='a'+str(len(arrays));arrays[key]=np.ascontiguousarray(array) if array.ndim else array
        return {'array':key,'tensor':tensor,'scalar':not tensor and not isinstance(item,np.ndarray)}
    tree=pack(value);arrays['__tree__']=np.frombuffer(json_bytes(tree),dtype=np.uint8)
    stream=io.BytesIO();np.savez_compressed(stream,**arrays)
    return stream.getvalue()


def decode(payload):
    with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
        tree=json.loads(archive['__tree__'].tobytes())
        def unpack(node):
            if 'none' in node:return None
            if 'dict' in node:return {unpack(k):unpack(v) for k,v in node['dict']}
            if 'tuple' in node:return tuple(unpack(v) for v in node['tuple'])
            if 'list' in node:return [unpack(v) for v in node['list']]
            array=np.array(archive[node['array']],copy=True)
            if node['tensor']:return torch.from_numpy(array)
            return array.item() if node['scalar'] else array
        return unpack(tree)


def identity(split,partition,protocol,protocol_sha,manifest):
    return {'schema_version':1,'split_id':split.split_id,'split_file_sha256':split.sha256,
            'geometry_sha256':split.geometry_sha256,'partition':partition,
            'protocol_id':protocol['protocol_id'],'protocol_sha256':protocol_sha,
            'data_manifest_sha256':sha256(manifest),'data_verified':True,
            'channel_scales_sha256':sha256(Path(manifest).parent.parent/'references'/split.split_id/'channel_scales.json')}


def cache_path(storage,split,partition,protocol,protocol_sha):
    return storage.root/'references'/split.split_id/protocol['protocol_id']/protocol_sha/('truth_'+partition)


class TruthCache:
    def __init__(self,path,*,building=False,expected=None):
        self.path=Path(path);self.building=building;self.entries={};self.manifest=None
        database=self.path/'quantities.sqlite3'
        if building:
            self.connection=sqlite3.connect(database)
            self.connection.execute('PRAGMA journal_mode=OFF')
            self.connection.execute('CREATE TABLE quantities (key TEXT PRIMARY KEY, sha256 TEXT NOT NULL, payload BLOB NOT NULL)')
        else:
            record=json.loads((self.path/'manifest.json').read_text())
            schema_validate(record,'truth_cache.schema.json')
            if expected is not None and any(record.get(k)!=v for k,v in expected.items()):
                raise ValueError('Truth cache identity differs from split/protocol/checksum manifest')
            if database.is_symlink() or sha256(database)!=record['cache_sha256']:
                raise ValueError('Truth cache file SHA256 mismatch')
            self.connection=sqlite3.connect(database.as_uri()+'?mode=ro&immutable=1',uri=True)
            self.entries={key:digest for key,digest in self.connection.execute('SELECT key,sha256 FROM quantities')}
            if self.entries!=record['entries']:raise ValueError('Truth cache entry manifest differs')
            self.manifest=record

    def value(self,kind,key,calculate=None):
        encoded=json.dumps([kind,key],sort_keys=True,separators=(',',':'),allow_nan=False)
        row=self.connection.execute('SELECT sha256,payload FROM quantities WHERE key=?',(encoded,)).fetchone()
        if row is None:
            if not self.building or calculate is None:
                raise ValueError('Required verified truth-cache entry is missing: '+encoded)
            payload=encode(calculate());digest=hashlib.sha256(payload).hexdigest()
            self.connection.execute('INSERT INTO quantities VALUES (?,?,?)',(encoded,digest,payload))
            self.entries[encoded]=digest
        else:
            digest,payload=row
            if hashlib.sha256(payload).hexdigest()!=digest or self.entries.get(encoded)!=digest:
                raise ValueError('Truth cache entry SHA256 mismatch: '+encoded)
        return decode(payload)

    def finish(self,storage,expected,source_commit=None):
        if not self.building:raise ValueError('Cannot finalize a read-only cache')
        self.connection.commit();self.connection.close();self.connection=None
        record={**expected,'cache_sha256':sha256(self.path/'quantities.sqlite3'),
                'entries':dict(sorted(self.entries.items())),'source_commit':source_commit}
        schema_validate(record,'truth_cache.schema.json')
        with storage.new_file(self.path,'manifest.json') as stream:stream.write(json_bytes(record))
        self.manifest=record;return record

    def close(self):
        if self.connection is not None:self.connection.close();self.connection=None


def truth_value(ctx,kind,key,calculate):
    cache=getattr(ctx,'truth_cache',None)
    if cache is not None and cache.building and ctx.reader.hashes is None:
        raise ValueError('Truth caches require verified per-file checksums')
    return calculate() if cache is None else cache.value(kind,key,calculate)


def truth_dump_id(ctx,value):
    if isinstance(value,torch.Tensor):return getattr(value,'_gval_dump_id',None)
    return ctx.reader.lookup(value)


def open_cache(storage,split,partition,protocol,protocol_sha,*,required=True):
    path=cache_path(storage,split,partition,protocol,protocol_sha)
    if not path.exists() and not required:return None
    expected=identity(split,partition,protocol,protocol_sha,storage.root/'checksums'/(split.split_id+'.sha256'))
    return TruthCache(path,expected=expected)
