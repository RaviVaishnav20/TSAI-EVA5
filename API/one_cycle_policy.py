## STEP 1
# Different types of annealing functions
# AnnealFunc = Callable[[Number,Number,float], Number]
import numpy as np
import functools
from functools import partial, reduce
from typing import Any, Collection, Callable, NewType, List, Union, TypeVar, Optional, Tuple
from numbers import Number
# export

AnnealFunc = Callable[[Number, Number, float], Number]


def annealing_no(start: Number, end: Number, pct: float) -> Number:
    "No annealing, always return `start`"
    return start


def annealing_linear(start: Number, end: Number, pct: float) -> Number:
    "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0"
    return start + pct * (end - start)


def annealing_exp(start: Number, end: Number, pct: float) -> Number:
    "Exponentially anneal from `start` to `end` as pct goes from 0.0 to 1.0"
    return start * (end / start) ** pct


def annealing_cos(start: Number, end: Number, pct: float) -> Number:
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0"
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


def do_annealing_poly(start: Number, end: Number, pct: float, degree: Number) -> Number:
    "Helper function for `anneal_poly`"
    return end + (start - end) * (1 - pct) ** degree


def annealing_poly(degree: Number) -> Number:
    "Anneal polynomically from `start` to `end` as pct goes from 0.0 to 1.0"
    return functools.partial(do_annealing_poly, degree=degree)


def is_tuple(x: Any) -> bool:
    return isinstance(x, tuple)


StartOptEnd = Union[float, Tuple[float, float]]


# def is_tuple(x):
#   return isinstance(x, tuple)

class Stepper():
    "Used to \"step\" from start,end (`vals`) over `n_iter` iterations on a schedule defined by `func` (defaults to linear)"

    def __init__(self, vals: StartOptEnd, n_iter: int, func: Optional[AnnealFunc] = None):
        self.start, self.end = (vals[0], vals[1]) if is_tuple(vals) else (vals, 0)
        self.n_iter = n_iter
        if func is None:
            self.func = annealing_linear if is_tuple(vals) else annealing_no
        else:
            self.func = func
        self.n = 0

    def step(self) -> Number:
        "Return next value along annealed schedule"
        self.n += 1
        return self.func(self.start, self.end, self.n / self.n_iter)

    @property
    def is_done(self) -> bool:
        "Schedule completed"
        return self.n >= self.n_iter


class Stepper():
    "Used to \"step\" from start,end (`vals`) over `n_iter` iterations on a schedule defined by `func` (defaults to linear)"

    def __init__(self, vals: StartOptEnd, n_iter: int, func: Optional[AnnealFunc] = None):
        self.start, self.end = (vals[0], vals[1]) if is_tuple(vals) else (vals, 0)
        self.n_iter = n_iter
        if func is None:
            self.func = annealing_linear if is_tuple(vals) else annealing_no
        else:
            self.func = func
        self.n = 0

    def step(self) -> Number:
        "Return next value along annealed schedule"
        self.n += 1
        return self.func(self.start, self.end, self.n / self.n_iter)

    @property
    def is_done(self) -> bool:
        "Schedule completed"
        return self.n >= self.n_iter


class OneCycleScheduler():
    def __init__(self, model, trainloader, optimizer, loss_fn, lr_max=0.1, moms=(0.95, 0.85),
                 div_factor=25.0, pct_start=0.5):
        self.opt = optimizer
        self.lr = optimizer.param_groups[-1]["lr"]
        self.momentum = optimizer.param_groups[-1]["momentum"]
        # self.weight_decay = optimizer.param_groups[-1]["weight_decay"]

        self.trainloader = trainloader

        self.lr_max = lr_max
        self.moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start

    def steps(self, StartOptEnd):
        "Build anneal schedule for all of the parameters"
        return [Stepper(step, n_iter, func=func) for (step, (n_iter, func)) in zip(StartOptEnd, self.phases)]

    # def steps(self, StartEnd):
    #   all_par_shedule = [Stepper(step, n_iter, annealing_func=func) for (step, (n_iter, func)) in zip (StartEnd, self.phases)]
    #   return all_par_shedule

    def on_train_begain(self, epochs: int, **kwargs: Any) -> None:
        n = len(self.trainloader) * (epochs)
        a1 = int(n * self.pct_start)
        a2 = n - a1

        self.phases = ((a1, annealing_linear), (a2, annealing_linear))
        # self.phases = (annealing_linear, annealing_cos,annealing_no)
        lr_min = self.lr_max / self.div_factor

        self.lr_scheds = self.steps(((lr_min, self.lr_max), (self.lr_max, lr_min / 1e4)), )
        self.moms_scheds = self.steps((self.moms, (self.moms[1], self.moms[0])), )

        self.lr = self.lr_scheds[0].start
        self.momentum = self.moms_scheds[0].start

        self.idx_s = 0

        return self.lr, self.momentum

    def on_batch_end(self):
        if self.idx_s >= len(self.lr_scheds):
            return True
        self.lr = self.lr_scheds[self.idx_s].step()
        self.momentum = self.moms_scheds[self.idx_s].step()
        # print(self.lr_scheds[self.idx_s].is_done)

        if self.lr_scheds[self.idx_s].is_done:
            self.idx_s += 1

        return self.lr, self.momentum