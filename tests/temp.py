from monograd import tensor, ops

a = tensor.Tensor([1, 2], op=ops.LOADOP())
a.op(tensor.Tensor([2, 4], op=ops.LOADOP()))

