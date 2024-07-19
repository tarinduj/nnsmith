from __future__ import annotations

import logging
import os
import pickle
from abc import ABC
from os import PathLike
from typing import Callable, Dict, List, Type

import jax
import jax.numpy as jnp
import numpy as np
from multipledispatch import dispatch

from nnsmith.abstract.op import AbsOpBase, AbsTensor
from nnsmith.gir import GraphIR
from nnsmith.materialize import Model, Oracle
from nnsmith.materialize.jax.forward import ALL_JAX_OPS
from nnsmith.materialize.jax.jaxnet import JAXNet
from nnsmith.util import register_seed_setter

JAXNetCallable = Callable[..., Dict[str, jnp.ndarray]]

@dispatch(dict)
def randn_from_specs(specs: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    return {
        name: jax.random.normal(jax.random.PRNGKey(0), shape=spec.shape).astype(spec.dtype)
        for name, spec in specs.items()
    }

def np_dict_from_jax(x: Dict[str, jnp.ndarray]) -> Dict[str, np.ndarray]:
    return {key: np.array(value) for key, value in x.items()}

def jax_dict_from_np(x: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
    return {key: jnp.array(value) for key, value in x.items()}

class JAXModel(Model, ABC):
    def __init__(self, ir: GraphIR) -> None:
        super().__init__()
        self.ir = ir
        self.net = JAXNet(ir=ir)

    @classmethod
    def from_gir(cls: Type["JAXModel"], ir: GraphIR, **kwargs) -> "JAXModel":
        return cls(ir)

    @property
    def version(self) -> str:
        return jax.__version__

    @property
    def native_model(self) -> JAXNet:
        return self.net

    @property
    def input_like(self) -> Dict[str, AbsTensor]:
        return {ik: self.ir.vars[ik] for ik in self.ir.input_var()}

    @property
    def output_like(self) -> Dict[str, AbsTensor]:
        return {ok: self.ir.vars[ok] for ok in self.ir.leaf_var()}

    @property
    def input_specs(self) -> Dict[str, jnp.ndarray]:
        return {k: jnp.zeros(aten.shape, dtype=aten.dtype.numpy()) for k, aten in self.input_like.items()}

    @staticmethod
    def name_suffix() -> str:
        return ""

    def make_oracle(self, inputs: Dict[str, jnp.ndarray] = None) -> Oracle:
        if inputs is None:
            input_dict = self.random_inputs()
        else:
            input_dict = inputs
        output_dict = self.run(input_dict)

        input_dict = np_dict_from_jax(input_dict)
        output_dict = np_dict_from_jax(output_dict)

        return Oracle(input_dict, output_dict, provider="jax")

    def dump(self, path: PathLike = "saved_jaxmodel") -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, JAXModel.gir_name()), "wb") as f:
            pickle.dump(self.ir, f)

    def dump_with_oracle(self, path: PathLike, inputs: Dict[str, jnp.ndarray] = None) -> None:
        self.dump(path)
        oracle = self.make_oracle(inputs)
        oracle.dump(os.path.join(path, Oracle.name()))

    @classmethod
    def load(cls, path: PathLike) -> "JAXModel":
        with open(os.path.join(path, cls.gir_name()), "rb") as f:
            gir: GraphIR = pickle.load(f)
        return cls(gir)

    @staticmethod
    def gir_name() -> str:
        return "gir.pkl"

    def random_inputs(self) -> Dict[str, jnp.ndarray]:
        return {
            name: jax.random.normal(jax.random.PRNGKey(0), shape=spec.shape).astype(spec.dtype)
            for name, spec in self.input_specs.items()
        }

    def run(self, inputs: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        return self.net(**inputs)

    def refine_weights(self) -> None:
        pass

    @staticmethod
    def operators() -> List[Type[AbsOpBase]]:
        return list(ALL_JAX_OPS)

    @property
    def import_libs(self) -> List[str]:
        return ["import jax", "import jax.numpy as jnp"]

    @staticmethod
    def add_seed_setter() -> None:
        register_seed_setter("jax", lambda seed: jax.random.PRNGKey(seed), overwrite=True)

class JAXModelCPU(JAXModel):
    def __init__(self, ir: GraphIR) -> None:
        super().__init__(ir)
        jax.config.update('jax_platform_name', 'cpu')

class JAXModelGPU(JAXModel):
    def __init__(self, ir: GraphIR) -> None:
        super().__init__(ir)
        jax.config.update('jax_platform_name', 'gpu')