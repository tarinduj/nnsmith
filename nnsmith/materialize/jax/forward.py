from functools import partial
from typing import List, Type

import jax
import jax.numpy as jnp

from nnsmith.abstract.op import *
from nnsmith.materialize import framework_operator_impl
from nnsmith.materialize.jax.dialect import *

JAX_REALIZABLE_OPS = FULL_OPERATOR_SETS["core"] + FULL_OPERATOR_SETS["jax"]
ALL_JAX_OPS: List[Type[AbsOpBase]] = []

operator_impl = partial(framework_operator_impl, JAX_REALIZABLE_OPS, ALL_JAX_OPS)

@operator_impl(Constant)
def forward_fn(op: Constant):
    dtype = op.abs_tensor.dtype.numpy()
    data = jax.random.normal(jax.random.PRNGKey(0), op.abs_tensor.shape).astype(dtype)
    return lambda: jnp.array(data, dtype=dtype)

@operator_impl(ReLU)
def forward_fn(op: ReLU):
    return jax.nn.relu

@operator_impl(GELU)
def forward_fn(op: GELU):
    return jax.nn.gelu

@operator_impl(LeakyReLU)
def forward_fn(op: LeakyReLU):
    return jax.nn.leaky_relu

@operator_impl(Sigmoid)
def forward_fn(op: Sigmoid):
    return jax.nn.sigmoid

@operator_impl(Cos)
def forward_fn(op: Cos):
    return jnp.cos

@operator_impl(Asin)
def forward_fn(op: Asin):
    return jnp.arcsin

@operator_impl(Acos)
def forward_fn(op: Acos):
    return jnp.arccos

@operator_impl(Tan)
def forward_fn(op: Tan):
    return jnp.tan

@operator_impl(Atan)
def forward_fn(op: Atan):
    return jnp.arctan

@operator_impl(Abs)
def forward_fn(op: Abs):
    return jnp.abs

@operator_impl(Where)
def forward_fn(op: Where):
    return jnp.where

@operator_impl(Add)
def forward_fn(op: Add):
    return jnp.add

@operator_impl(Sub)
def forward_fn(op: Sub):
    return jnp.subtract

@operator_impl(Mul)
def forward_fn(op: Mul):
    return jnp.multiply

@operator_impl(Div)
def forward_fn(op: Div):
    return jnp.divide

@operator_impl(Max)
def forward_fn(op: Max):
    return jnp.maximum

@operator_impl(Min)
def forward_fn(op: Min):
    return jnp.minimum

@operator_impl(Equal)
def forward_fn(op: Equal):
    return jnp.equal

@operator_impl(Greater)
def forward_fn(op: Greater):
    return jnp.greater

@operator_impl(Less)
def forward_fn(op: Less):
    return jnp.less

@operator_impl(And)
def forward_fn(op: And):
    return jnp.logical_and

@operator_impl(Or)
def forward_fn(op: Or):
    return jnp.logical_or

@operator_impl(Xor)
def forward_fn(op: Xor):
    return jnp.logical_xor

@operator_impl(Pow)
def forward_fn(op: Pow):
    return jnp.power

@operator_impl(Floor)
def forward_fn(op: Floor):
    return jnp.floor

@operator_impl(Ceil)
def forward_fn(op: Ceil):
    return jnp.ceil

@operator_impl(Clip)
def forward_fn(op: Clip):
    if op.input_like[0].dtype in DTYPE_GEN_FLOATS:
        return lambda x: jnp.clip(x, -1.5, 1.5)
    else:
        return lambda x: jnp.clip(x, -1, 1)

@operator_impl(Round)
def forward_fn(op: Round):
    return jnp.round

@operator_impl(Sqrt)
def forward_fn(op: Sqrt):
    return jnp.sqrt

@operator_impl(Log2)
def forward_fn(op: Log2):
    return jnp.log2

@operator_impl(Neg)
def forward_fn(op: Neg):
    return jnp.negative

@operator_impl(Softmax)
def forward_fn(op: Softmax):
    return lambda x: jax.nn.softmax(x, axis=op.dim)

@operator_impl(Slice)
def forward_fn(op: Slice):
    def _slice(x):
        slices = [slice(None)] * x.ndim
        slices[op.extra_attrs["axis"]] = slice(op.start, op.end, op.step)
        return x[tuple(slices)]
    return _slice

@operator_impl(BatchNorm2d)
def forward_fn(op: BatchNorm2d):
    return lambda x: jax.nn.batch_norm(x, None, None, use_running_average=False, axis=1)

@operator_impl(Reshape)
def forward_fn(op: Reshape):
    return lambda x: jnp.reshape(x, op.target_shape)

@operator_impl(Transpose)
def forward_fn(op: Transpose):
    def _transpose(x):
        aten = op.input_like[0]
        dim0, dim1 = op._init_swap_dims(aten.shape)
        perm = list(range(aten.ndims))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return jnp.transpose(x, perm)
    return _transpose

@operator_impl(Dense)
def forward_fn(op: Dense):
    return lambda x: jax.nn.dense(x, jnp.ones((op.ifeat, op.ofeat)))

@operator_impl(Squeeze)
def forward_fn(op: Squeeze):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: jnp.squeeze(x, axis=op.extra_attrs["reduce_dim"])
    return jnp.squeeze

@operator_impl(Unsqueeze)
def forward_fn(op: Unsqueeze):
    return lambda x: jnp.expand_dims(x, axis=op.extra_attrs["expand_dim"])

@operator_impl(ReduceSum)
def forward_fn(op: ReduceSum):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: jnp.sum(x, axis=op.extra_attrs["reduce_dim"])
    return jnp.sum

@operator_impl(ReduceMin)
def forward_fn(op: ReduceMin):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: jnp.min(x, axis=op.extra_attrs["reduce_dim"])
    return jnp.min

@operator_impl(ReduceMax)
def forward_fn(op: ReduceMax):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: jnp.max(x, axis=op.extra_attrs["reduce_dim"])
    return jnp.max

@operator_impl(ReduceMean)
def forward_fn(op: ReduceMean):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: jnp.mean(x, axis=op.extra_attrs["reduce_dim"])
    return jnp.mean

@operator_impl(ReduceProd)
def forward_fn(op: ReduceProd):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: jnp.prod(x, axis=op.extra_attrs["reduce_dim"])
    return jnp.prod

@operator_impl(ArgMin)
def forward_fn(op: ArgMin):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: jnp.argmin(x, axis=op.extra_attrs["reduce_dim"])
    return jnp.argmin

@operator_impl(ArgMax)
def forward_fn(op: ArgMax):
    if op.extra_attrs["reduce_dim"] is not None:
        return lambda x: jnp.argmax(x, axis=op.extra_attrs["reduce_dim"])
    return jnp.argmax

@operator_impl(Tril)
def forward_fn(op: Tril):
    return lambda x: jnp.tril(x, k=op.diagonal)

@operator_impl(Triu)
def forward_fn(op: Triu):
    return lambda x: jnp.triu(x, k=op.diagonal)

@operator_impl(Concat)
def forward_fn(op: Concat):
    axis = op.extra_attrs["axis"]
    return lambda *args: jnp.concatenate(args, axis=axis)

@operator_impl(Cast)
def forward_fn(op: Cast):
    return lambda x: x.astype(op.extra_attrs["to"].numpy())

@operator_impl(JAXMatMul)
def forward_fn(op: JAXMatMul):
    return jnp.matmul

@operator_impl(Reverse)
def forward_fn(op: Reverse):
    return lambda x: jnp.flip(x, axis=op.extra_attrs["axis"])

# @operator_impl(Cholesky)
# def forward_fn(op: Cholesky):
#     return jnp.linalg.cholesky

# @operator_impl(Eigh)
# def forward_fn(op: Eigh):
#     return jnp.linalg.eigh

@operator_impl(Conv1d)
def forward_fn(op: Conv1d):
    return lambda x, w: jax.lax.conv(x, w, (op.stride,), op.extra_attrs["padding"])

@operator_impl(MaxPool2d)
def forward_fn(op: MaxPool2d):
    return lambda x: jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, op.kh, op.kw, 1), (1, op.stride, op.stride, 1), op.extra_attrs["padding"])

@operator_impl(AvgPool2d)
def forward_fn(op: AvgPool2d):
    def pool(x):
        pooled = jax.lax.reduce_window(x, 0., jax.lax.add, (1, op.kh, op.kw, 1), (1, op.stride, op.stride, 1), op.extra_attrs["padding"])
        return pooled / (op.kh * op.kw)
    return pool

# @operator_impl(Gather)
# def forward_fn(op: Gather):
#     return lambda params, indices: jnp.take(params, indices, axis=op.extra_attrs["axis"])

@operator_impl(PReLU)
def forward_fn(op: PReLU):
    return lambda x: jnp.where(x > 0, x, 0.01 * x)