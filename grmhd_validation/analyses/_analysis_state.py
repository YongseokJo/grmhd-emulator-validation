"""Explicit per-analysis compatibility configuration, without model/data globals."""
from collections.abc import Mapping
from contextvars import ContextVar

CURRENT = ContextVar('gval_analysis_state')


def current():
    return CURRENT.get()


def setting(name):
    return current()['ctx'].protocol['constants']['full'][name]


class PhysicsSettings(Mapping):
    def __getitem__(self, name):
        state=current();ctx=state['ctx']
        if name in ('floor_rho','floor_ug'):
            return ctx.constants['loss_space'][name]
        if name=='dt_dump':return ctx.split.dt_dump
        if name=='eos_gamma':return float(state['geometry'].eos_gamma)
        return ctx.constants['full'][name]

    def __iter__(self):
        return iter((*current()['ctx'].constants['full'],'floor_rho','floor_ug','eos_gamma','dt_dump'))

    def __len__(self):
        return len(tuple(iter(self)))


CONFIG=PhysicsSettings()
