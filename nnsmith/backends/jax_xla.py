from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
from multipledispatch import dispatch

from nnsmith.backends.factory import BackendCallable, BackendFactory
from nnsmith.materialize.jax import JAXModel, np_dict_from_jax, jax_dict_from_np

class JAXLA(BackendFactory):
    def __init__(self, target="cpu", optmax: bool = True):
        super().__init__(target, optmax)

        if self.target == "cpu":
            jax.config.update('jax_platform_name', 'cpu')
        elif self.target == "cuda":
            jax.config.update('jax_platform_name', 'gpu')
        else:
            raise ValueError(
                f"Unknown device: {self.target}. Only `cpu` and `cuda` are supported."
            )

    @property
    def system_name(self) -> str:
        return "jax_xla"

    @property
    def version(self) -> str:
        return jax.__version__

    @property
    def import_libs(self) -> List[str]:
        return ["import jax", "import jax.numpy as jnp"]

    @dispatch(JAXModel)
    def make_backend(self, model: JAXModel) -> BackendCallable:
        @jax.jit
        def compiled_model(**inputs):
            return model.net(**inputs)

        def closure(inputs: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
            result = np_dict_from_jax(compiled_model(**jax_dict_from_np(inputs)))
            return result

        return closure

    def emit_compile(self, opt_name: str, mod_name: str, inp_name: Optional[str] = None) -> str:
        return f"{opt_name} = jax.jit({mod_name})"

    def emit_run(self, out_name: str, opt_name: str, inp_name: str) -> str:
        return f"{out_name} = {opt_name}(**{inp_name})"