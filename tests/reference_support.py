"""Runtime-only access to Kevin Fortune/Sachin's pinned reference definitions.

Source: main@ebcc218, Fall26/eval/evaluate_forecast_ds.py and common/*.py;
TRON model: fall26-tron-rc300@2667a5f, Fall26/tron/model.py.
This test helper never writes to the source checkout or vendors original code.
"""
import argparse
import ast
import copy
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REFERENCE=Path(os.environ.get('GVAL_REFERENCE_CHECKOUT','/u/gkerex/project/reference/neural_operator_msai_spring26'))


def source(path, commit='ebcc218'):
    return subprocess.run(['git','-C',str(REFERENCE),'show',commit+':'+path],
                          check=True,capture_output=True,text=True).stdout


def definitions(text, namespace, filename, *, assignments=False):
    tree=ast.parse(text)
    nodes=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):nodes.append(node)
        elif assignments and isinstance(node,ast.Assign):
            try:ast.literal_eval(node.value)
            except (ValueError,TypeError):continue
            nodes.append(node)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,'exec'),namespace)
    return tree


def load_reference():
    namespace=dict(np=np,torch=torch,nn=nn,F=F,os=os,sys=sys,time=time,
                   warnings=warnings,plt=plt,matplotlib=matplotlib,Line2D=Line2D,argparse=argparse,
                   logger=logging.getLogger('gval.reference'))
    settings={}
    exec(compile(source('Fall26/common/settings.py'),'<reference settings>','exec'),settings)
    namespace['CONFIG']=settings['PHYSICS'].copy()
    from torch.utils.checkpoint import checkpoint
    namespace['checkpoint']=checkpoint
    definitions(source('Fall26/common/normalizer.py'),namespace,'<reference normalizer>')
    definitions(source('Fall26/common/physics_loss.py'),namespace,'<reference physics>')
    definitions(source('Fall26/tron/model.py','2667a5f'),namespace,'<reference TRON model>')
    evaluator=source('Fall26/eval/evaluate_forecast_ds.py')
    tree=definitions(evaluator,namespace,'<reference evaluator>',assignments=True)
    # Keep each module's original function globals together, as the reference
    # wrapper does when it installs common's normalizer and physics settings.
    return namespace,tree


def numeric_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().numpy().copy()
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,dict):return {str(key):numeric_copy(v) for key,v in value.items()}
    if isinstance(value,(tuple,list)):return [numeric_copy(v) for v in value]
    return copy.deepcopy(value)


def observe_return(function,*args,loop_fields=None,tree=None):
    """Observe original numeric locals without replacing any computation."""
    captured={}
    previous=sys.gettrace();rows=[];line=None
    if loop_fields:
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function.__name__)
        variables=set(loop_fields.values())
        for loop in ast.walk(node):
            if not isinstance(loop,ast.For):continue
            assigned={n.id for n in ast.walk(loop) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
            if variables<=assigned:line=loop.lineno;break
        if line is None:raise AssertionError('No complete source observation loop: '+function.__name__)
    def observer(frame,event,arg):
        if frame.f_code is function.__code__:
            if event=='return':captured.update(frame.f_locals)
            if event=='line' and frame.f_lineno==line and all(v in frame.f_locals for v in loop_fields.values()):
                rows.append({key:numeric_copy(frame.f_locals[var]) for key,var in loop_fields.items()})
            return observer
        return None
    try:
        sys.settrace(observer)
        result=function(*args)
    finally:sys.settrace(previous)
    captured['_observed_iterations']=rows
    return result,captured
