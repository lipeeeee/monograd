import numpy as np
from tests.core import Tester
from monograd.tensor import Tensor
from monograd.ops import ADD, MUL

class Test_Ops(Tester):
    def __init__(self):
        super().__init__()

    def test_all(self):
        super().test_fn(self.test_add_forward)
        super().test_fn(self.test_add_backward)
        super().test_fn(self.test_mul_forward)
        super().test_fn(self.test_mul_backward)

        super().test_fn(self.test_matmul_forward)
        super().test_fn(self.test_matmul_backward)
        super().test_fn(self.test_transpose_forward)
        super().test_fn(self.test_transpose_backward)

        super().test_fn(self.test_relu_forward)
        super().test_fn(self.test_relu_backward)
        super().test_fn(self.test_sum_forward)
        super().test_fn(self.test_sum_backward)

    def test_add_forward(self):
        a = Tensor(1)
        b = a + 1
        assert isinstance(b, Tensor), f"Result of ADDOP expected type Tensor, got {type(b)}" 
        assert b.data == 2, f"ADD result error: expected 2 got {b.data}"

        a = Tensor([1, 2])
        b = Tensor([2, 4])
        c = a + b 
        assert isinstance(c, Tensor), f"Result of ADDOP expected type Tensor, got {type(b)}"
        assert c.data[0] == 3, f"ADD result error: expected 3 got {c.data}"
        assert c.data[1] == 6, f"ADD result error: expected 6 got {c.data}" 

        a = Tensor(1)
        b = a + 0
        assert isinstance(b, Tensor), f"Result of ADDOP expected type Tensor, got {type(b)}"
        assert a.data == b.data, "ADD result error"

        a = ADD.apply(Tensor(1), Tensor(2)) # call op itself
        assert a.data == 3, f"ADD result error: expected 3 got {a.data}"

    def test_add_backward(self):
        a = Tensor(2.0, requires_grad=True)
        b = Tensor(3.0, requires_grad=True)
        c = a + b
        c.backward()
        assert isinstance(a.grad, Tensor), f"Gradient type expected to be Tensor, got {type(a.grad)}"
        assert isinstance(b.grad, Tensor), f"Gradient type expected to be Tensor, got {type(a.grad)}"
        assert a.grad.data == 1.0, f"ADD backward error: expected 1.0, got {a.grad}"
        assert b.grad.data == 1.0, f"ADD backward error: expected 1.0, got {b.grad}"

        a = Tensor(3.0, requires_grad=True)
        b = a + a
        b.backward()
        assert isinstance(a.grad, Tensor), f"Gradient type expected to be Tensor, got {type(a.grad)}"
        assert a.grad.data == 2.0, f"ADD backward error: expected 2.0, got {a.grad.data}"

    def test_mul_forward(self):
        a = Tensor(1, requires_grad=True)
        b = a * 2
        assert isinstance(b, Tensor), f"Result of MULOP expected to type Tensor, instead got: {type(b)}"
        assert b.data == a.data * 2, "MUL result error"

        a = Tensor([1, 2])
        b = Tensor([2, 4])
        c = a * b 
        assert isinstance(c, Tensor), f"Result of MULOP expected to type Tensor, instead got: {type(c)}"
        assert c.data[0] == 2, f"MUL result error: expected 2 got {c.data[0]}"
        assert c.data[1] == 8, f"MUL result error: expected 8 got {c.data[1]}"

        a = Tensor(1)
        b = a * 1
        assert isinstance(b, Tensor), f"Result of MULOP expected to type Tensor, instead got: {type(b)}"
        assert a.data == b.data, "MUL result error"

        a = MUL.apply(Tensor(1), Tensor(2)) # call op itself
        assert a.data == 2, f"MUL result error: expected 2 got {a.data}"

    def test_mul_backward(self):
        a = Tensor(2.0, requires_grad=True)
        b = Tensor(4.0, requires_grad=True)
        c = a * b
        c.backward()
        assert isinstance(a.grad, Tensor), f"Gradient type expected to be Tensor, got {type(a.grad)}"
        assert isinstance(b.grad, Tensor), f"Gradient type expected to be Tensor, got {type(a.grad)}"
        assert a.grad.data == 4.0, f"MUL backward error: expected 4.0, got {a.grad.data}"
        assert b.grad.data == 2.0, f"MUL backward error: expected 2.0, got {b.grad.data}"

        a = Tensor(3.0)
        b = a * a
        b.backward()
        # dy/dx = 2x = 6.0
        assert isinstance(a.grad, Tensor), f"Gradient type expected to be Tensor, got {type(a.grad)}"
        assert a.grad.data == 6.0, f"MUL backward error: expected 6.0, got {a.grad.data}"

    def test_matmul_forward(self):
        a = Tensor([[1.0, 2.0]])
        b = Tensor([[3.0], [4.0]])
        c = a @ b
        assert isinstance(c, Tensor), f"Result of MATMUL expected type Tensor, got {type(c)}"
        assert c.shape == (1, 1), f"MATMUL shape error: expected (1, 1), got {c.shape}"
        # 1*3 + 2*4 = 3 + 8 = 11
        assert c.data[0][0] == 11.0, f"MATMUL result error: expected 11.0, got {c.data}"

        # Identity Matrix Check
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        identity = Tensor([[1.0, 0.0], [0.0, 1.0]])
        c = a @ identity
        assert np.array_equal(c.data, a.data), f"MATMUL identity error"

    def test_matmul_backward(self):
        x = Tensor(np.ones((2, 3)), requires_grad=True)
        w = Tensor(np.ones((3, 2)), requires_grad=True)
        out = x @ w
        out.backward()

        assert isinstance(x.grad, Tensor), f"MATMUL backward X gradient was not calculated for X"
        assert isinstance(w.grad, Tensor), f"MATMUL backward W gradient was not calculated for W"
        assert x.grad.data.shape == (2, 3), f"MATMUL backward X shape error: got {x.grad.shape}"
        assert w.grad.shape == (3, 2), f"MATMUL backward W shape error: got {w.grad.shape}"

        # dL/dx = grad_out @ w.T -> Ones(2,2) @ Ones(2,3) = Each elem is sum(col) = 2.0
        # dL/dw = x.T @ grad_out -> Ones(3,2) @ Ones(2,2) = Each elem is sum(row) = 2.0
        assert np.all(x.grad.data == 2.0), f"MATMUL backward value error (x)"
        assert np.all(w.grad.data == 2.0), f"MATMUL backward value error (w)"

    def test_transpose_forward(self):
        a = Tensor([[1.0, 2.0, 3.0]])
        b = a.T
        assert b.shape == (3, 1), f"TRANSPOSE shape error: expected (3, 1), got {b.shape}"
        assert b.data[0][0] == 1.0 and b.data[2][0] == 3.0

        c = a.transpose()
        assert c.shape == (3, 1)

    def test_transpose_backward(self):
        # y = x.T
        # if dy/dy = 1, then dy/dx = 1 (just transposed back)
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = a.T
        b.backward()

        # Gradient should be all ones, matching shape of a (2, 2)
        assert a.grad, f"TRANSPOSE backward was not calculated for A"
        assert a.grad.shape == (2, 2)
        assert np.all(a.grad.data == 1.0)

    def test_relu_forward(self):
        a = Tensor([-1.0, 0.0, 1.0])
        b = a.relu()
        assert isinstance(b, Tensor), f"RELU didn't return a Tensor obj"
        assert np.array_equal(b.data, [0.0, 0.0, 1.0]), f"RELU forward error: got {b.data}"

        a = Tensor([-5.0, -2.0])
        b = a.relu()
        assert isinstance(b, Tensor), f"RELU didn't return a Tensor obj"
        assert np.array_equal(b.data, [0.0, 0.0]), f"RELU forward error: got {b.data}"

    def test_relu_backward(self):
        a = Tensor([-1.0, 2.0], requires_grad=True)
        b = a.relu()
        assert isinstance(b, Tensor), f"RELU didn't return a Tensor obj"
        b.backward()
        assert isinstance(a.grad, Tensor), f"RELU backward error: didn't compute gradient of *a*"
        assert np.array_equal(a.grad.data, [0.0, 1.0]), f"RELU backward error: got {a.grad.data}"

    def test_sum_forward(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = a.sum()
        assert isinstance(b, Tensor)
        assert b.data == 6.0, f"SUM forward error: got {b.data}"

        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = a.sum(axis=0)
        assert np.array_equal(b.data, [4.0, 6.0]), f"SUM axis=0 error: got {b.data}"

        c = a.sum(axis=1)
        assert np.array_equal(c.data, [3.0, 7.0]), f"SUM axis=1 error: got {c.data}"

    def test_sum_backward(self):
        # Simple backward sum
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = a.sum()
        b.backward()
        assert isinstance(a.grad, Tensor), f"SUM backward error: didn't compute gradient of *a*"
        assert np.array_equal(a.grad.data, [1.0, 1.0, 1.0]), f"SUM backward error: got {a.grad.data}"

        # broadcasting logic
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.sum(axis=0)

        z = y * Tensor([2.0, 3.0])
        z = z.sum()
        z.backward()

        expected = [[2.0, 3.0], [2.0, 3.0]]
        assert isinstance(x.grad, Tensor), f"SUM backward error: didn't compute gradient of *x*"
        assert np.array_equal(x.grad.data, expected), f"SUM axis backward error:\n{x.grad.data}"

Test_Ops().test_all()
