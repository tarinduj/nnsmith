import random
from typing import List, Tuple, Union

from functools import reduce

import jax.numpy as jnp
import jax.lax as lax

from nnsmith.abstract.arith import *
from nnsmith.abstract.dtype import DTYPE_GEN_ALL, DTYPE_GEN_FLOATS, DType
from nnsmith.abstract.op import (
    AbsOpBase,
    BcastBinaryOp,
    BinaryOpBase,
    ElementWiseUnaryOp,
    MatMul,
    UnaryOpBase,
    mark_materialize,
    rank_all,
    rank_from,
)
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.error import ConstraintCheck

@mark_materialize("jax")
class Dense(UnaryOpBase):
    in_dtypes = [(DType.float32,), (DType.float64,)]
    out_dtypes = [(DType.float32,), (DType.float64,)]

    def __init__(self, ifeat: Union[int, z3.ExprRef], ofeat: Union[int, z3.ExprRef]):
        super().__init__()
        self.ifeat = ifeat
        self.ofeat = ofeat
        self.inp_ranks = [rank_from(2)]
        self.out_ranks = [rank_from(2)]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        assert len(input_shapes) == 1, "Dense only takes one input, but got {}".format(
            len(input_shapes)
        )
        return [
            AbsTensor(
                shape=[*input_shapes[0].shape[:-1], self.ofeat],
                dtype=input_shapes[0].dtype,
            )
        ]

    def requires(self, input_shapes: List[AbsTensor]) -> List[z3.ExprRef]:
        ConstraintCheck.true(input_shapes[0].ndims >= 2)
        return [
            nnsmith_ge(self.ifeat, 1),
            nnsmith_ge(self.ofeat, 1),
            nnsmith_eq(input_shapes[0].shape[-1], self.ifeat),
        ]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [(out_abs_tensor[0].ndims, out_abs_tensor[0].dtype)]

@mark_materialize("jax")
class Conv2d(UnaryOpBase):
    in_dtypes = [(DType.float32,)]
    out_dtypes = [(DType.float32,)]

    def __init__(
        self,
        in_channels: Union[int, z3.ExprRef],
        out_channels: Union[int, z3.ExprRef],
        kernel_size: Union[int, z3.ExprRef],
        stride: Union[int, z3.ExprRef],
        padding: str,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.extra_attrs["padding"] = padding
        self.inp_ranks = [(4,)]
        self.out_ranks = [(4,)]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        ni, hi, wi, _ = input_shapes[0].shape
        if self.extra_attrs["padding"] == "VALID":
            ho = (hi - self.kernel_size) // self.stride + 1
            wo = (wi - self.kernel_size) // self.stride + 1
        elif self.extra_attrs["padding"] == "SAME":
            ho = (hi + self.stride - 1) // self.stride
            wo = (wi + self.stride - 1) // self.stride
        return [AbsTensor(shape=[ni, ho, wo, self.out_channels], dtype=input_shapes[0].dtype)]

    def requires(self, input_shapes):
        _, hi, wi, ci = input_shapes[0].shape
        return [
            nnsmith_eq(ci, self.in_channels),
            nnsmith_ge(self.out_channels, 1),
            nnsmith_ge(self.kernel_size, 1),
            nnsmith_ge(self.stride, 1),
            nnsmith_le(self.kernel_size, hi),
            nnsmith_le(self.kernel_size, wi),
        ]

@mark_materialize("jax")
class JAXMatMul(MatMul):
    def __init__(self):
        super().__init__()
        self.inp_ranks = [(2, 3), (2, 3)]
        self.out_ranks = [(2, 3)]

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        if out_abs_tensor[0].ndims == 2:
            return [
                (2, out_abs_tensor[0].dtype),
                (2, out_abs_tensor[0].dtype),
            ]
        ranks = [3, random.choice([2, 3])]
        random.shuffle(ranks)
        return [
            (ranks[0], out_abs_tensor[0].dtype),
            (ranks[1], out_abs_tensor[0].dtype),
        ]

@mark_materialize("jax")
class Reverse(UnaryOpBase):
    in_dtypes = [(i,) for i in DTYPE_GEN_ALL]
    out_dtypes = [(i,) for i in DTYPE_GEN_ALL]

    def __init__(self):
        super().__init__()
        self.inp_ranks = [rank_from(1)]
        self.out_ranks = [rank_from(1)]

    def _init_axis(self, input_shape: List[Union[int, z3.ExprRef]]):
        # axis is a list of integers
        # |axis| <= rank
        if "axis" not in self.extra_attrs:
            axis = []
            for i in range(len(input_shape)):
                if random.random() < 0.5:  # prob
                    axis.append(i)
            # TODO(@ganler): tflite crashes when axis is empty
            # remove this when tf fixes https://github.com/tensorflow/tensorflow/issues/62679
            axis = axis or [0]
            self.extra_attrs["axis"] = axis
        ConstraintCheck.le(len(self.extra_attrs["axis"]), len(input_shape))
        if self.extra_attrs["axis"]:
            ConstraintCheck.lt(max(self.extra_attrs["axis"]), len(input_shape))
        return self.extra_attrs["axis"]

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        _ = self._init_axis(input_shapes[0].shape)
        return input_shapes

    def requires(self, input_shapes):
        _ = self._init_axis(input_shapes[0].shape)
        return super().requires(input_shapes)

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        return [
            (out_abs_tensor[0].ndims, out_abs_tensor[0].dtype),
        ]

# Add more JAX-specific operators as needed