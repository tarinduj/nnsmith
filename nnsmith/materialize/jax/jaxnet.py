from dataclasses import dataclass
from typing import Callable, Dict, List, cast

import jax
import jax.numpy as jnp

from nnsmith.abstract.op import AbsOpBase, Input
from nnsmith.error import SanityCheck
from nnsmith.gir import GraphIR
from nnsmith.logging import JAX_LOG
from nnsmith.materialize.jax.forward import forward_fn

@dataclass
class Instr:
    fwd_fn: Callable
    inp_keys: List[str]
    out_keys: List[str]

class JAXNet:
    def __init__(self, ir: GraphIR) -> None:
        self.ir: GraphIR = ir
        self.instructions: List[Instr] = []

        for inst in self.ir.insts:
            if not isinstance(inst.iexpr.op, Input):
                op = cast(AbsOpBase, inst.iexpr.op)
                fwd_fn = forward_fn(op)
                SanityCheck.true(fwd_fn is not None, f"Bad impl for {inst.iexpr.op}")
                self.instructions.append(Instr(fwd_fn, inst.iexpr.args, inst.retvals()))

    def __call__(self, **kwargs) -> Dict[str, jnp.ndarray]:
        return self.__forward(**kwargs)

    def __forward(self, **kwargs) -> Dict[str, jnp.ndarray]:
        JAX_LOG.debug(f"Running with JIT")

        key2tensor: Dict[str, jnp.ndarray] = {}
        for key in self.ir.input_var():
            key2tensor[key] = kwargs[key]

        for instr in self.instructions:
            inp_tensors = [key2tensor[key] for key in instr.inp_keys]
            out_tensors = instr.fwd_fn(*inp_tensors)

            if isinstance(out_tensors, jnp.ndarray):
                out_tensors = [out_tensors]
            if isinstance(out_tensors, tuple):
                out_tensors = list(out_tensors)

            for i_out, out_key in enumerate(instr.out_keys):
                key2tensor[out_key] = out_tensors[i_out]

        return {k: key2tensor[k] for k in self.ir.leaf_var()}