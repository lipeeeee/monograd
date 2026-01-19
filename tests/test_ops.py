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
        super().test_fn(self.test_sub_forward)
        super().test_fn(self.test_sub_backward)
        super().test_fn(self.test_mul_forward)
        super().test_fn(self.test_mul_backward)

        super().test_fn(self.test_matmul_forward)
        super().test_fn(self.test_matmul_backward)
        super().test_fn(self.test_transpose_forward)
        super().test_fn(self.test_transpose_backward)
        super().test_fn(self.test_reshape_forward)
        super().test_fn(self.test_reshape_backward)

        super().test_fn(self.test_relu_forward)
        super().test_fn(self.test_relu_backward)
        super().test_fn(self.test_leakyrelu_forward)
        super().test_fn(self.test_leakyrelu_backward)
        super().test_fn(self.test_sum_forward)
        super().test_fn(self.test_sum_backward)
        
        super().test_fn(self.test_exp_forward)
        super().test_fn(self.test_exp_backward)
        super().test_fn(self.test_log_forward)
        super().test_fn(self.test_log_backward)

    def test_add_forward(self):
        a = Tensor(1)
        b = a + 1
        assert isinstance(b, Tensor), f"ADD forward error: OP didn't return Tensor object, instead got {type(b)}"
        assert b.data == 2, f"ADD forward compute error: expected 2 got {b.data}"

        a = Tensor([1, 2])
        b = Tensor([2, 4])
        c = a + b 
        assert isinstance(c, Tensor), f"ADD forward error: OP didn't return Tensor object, instead got {type(c)}"
        assert c.data[0] == 3, f"ADD forward compute error: expected 3 got {c.data}"
        assert c.data[1] == 6, f"ADD forward compute error: expected 6 got {c.data}" 

        a = Tensor(1)
        b = a + 0
        assert isinstance(b, Tensor), f"ADD forward error: OP didn't return Tensor object, instead got {type(b)}"
        assert a.data == b.data, f"ADD forward compute error: expected a == b and got {a.data} != {b.data}" 

        a = ADD.apply(Tensor(1), Tensor(2)) # call op itself
        assert a.data == 3, f"ADD forward compute error: expected 3 got {a.data}"

    def test_add_backward(self):
        a = Tensor(2.0, requires_grad=True)
        b = Tensor(3.0, requires_grad=True)
        c = a + b
        c.backward()
        assert isinstance(a.grad, Tensor), f"ADD backward error: didn't compute gradient of *a*({type(a.grad)})"
        assert isinstance(b.grad, Tensor), f"ADD backward error: didn't compute gradient of *b*({type(b.grad)})"
        assert a.grad.data == 1.0, f"ADD backward error: expected 1.0, got {a.grad}"
        assert b.grad.data == 1.0, f"ADD backward error: expected 1.0, got {b.grad}"

        a = Tensor(3.0, requires_grad=True)
        b = a + a
        b.backward()
        assert isinstance(a.grad, Tensor), f"ADD backward error: didn't compute gradient of *a*({type(b.grad)})"
        assert a.grad.data == 2.0, f"ADD backward error: expected 2.0, got {a.grad.data}"

    def test_sub_forward(self):
        a = Tensor(5)
        b = a - 2
        assert isinstance(b, Tensor), f"SUB forward error: OP didn't return Tensor object, instead got {type(b)}"
        assert b.data == 3, f"SUB forward compute error: expected 3 got {b.data}"

        a = Tensor([5, 4])
        b = Tensor([2, 1])
        c = a - b
        assert np.array_equal(c.data, [3, 3]), f"SUB forward compute error error: expected [3, 3] got {c.data}"

    def test_sub_backward(self):
        a = Tensor(5.0, requires_grad=True)
        b = Tensor(2.0, requires_grad=True)
        c = a - b
        c.backward()
        # dy/da = 1, dy/db = -1
        assert isinstance(a.grad, Tensor), f"SUB backward error: didn't compute gradient of *a*({type(a.grad)})"
        assert isinstance(b.grad, Tensor), f"SUB backward error: didn't compute gradient of *b*({type(b.grad)})"
        assert a.grad.data == 1.0, f"SUB backward error: expected 1.0 got {a.grad.data}"
        assert b.grad.data == -1.0, f"SUB backward error: expected -1.0 got {b.grad.data}"

    def test_mul_forward(self):
        a = Tensor(1, requires_grad=True)
        b = a * 2
        assert isinstance(b, Tensor), f"MUL forward error: OP didn't return Tensor object, instead got {type(b)}"
        assert b.data == a.data * 2, f"MUL forward compute error, expected b==a*2 instead got {b.data} != {a.data}*2"

        a = Tensor([1, 2])
        b = Tensor([2, 4])
        c = a * b 
        assert isinstance(c, Tensor), f"MUL forward error: OP didn't return Tensor object, instead got {type(c)}"
        assert c.data[0] == 2, f"MUL forward compute error: expected 2 got {c.data[0]}"
        assert c.data[1] == 8, f"MUL forward compute error: expected 8 got {c.data[1]}"

        a = Tensor(1)
        b = a * 1
        assert isinstance(b, Tensor), f"MUL forward error: OP didn't return Tensor object, instead got {type(b)}"
        assert a.data == b.data, f"MUL forward compute error: expected a == b instead got {a.data} != {b.data}"

        a = MUL.apply(Tensor(1), Tensor(2)) # call op itself
        assert a.data == 2, f"MUL forward compute error: expected a == 2 instead got {a.data} != 2"

    def test_mul_backward(self):
        a = Tensor(2.0, requires_grad=True)
        b = Tensor(4.0, requires_grad=True)
        c = a * b
        c.backward()
        assert isinstance(a.grad, Tensor), f"MUL backward error: didn't compute gradient of *a*({type(a.grad)})"
        assert isinstance(b.grad, Tensor), f"MUL backward error: didn't compute gradient of *b*({type(b.grad)})"
        assert a.grad.data == 4.0, f"MUL backward compute error: expected 4.0, got {a.grad.data}"
        assert b.grad.data == 2.0, f"MUL backward compute error: expected 2.0, got {b.grad.data}"

        a = Tensor(3.0)
        b = a * a
        b.backward()
        # dy/dx = 2x = 6.0
        assert isinstance(a.grad, Tensor), f"MUL backward error: didn't compute gradient of *a*({type(a.grad)})"
        assert a.grad.data == 6.0, f"MUL backward error: expected 6.0, got {a.grad.data}"

    def test_matmul_forward(self):
        a = Tensor([[1.0, 2.0]])
        b = Tensor([[3.0], [4.0]])
        c = a @ b
        assert isinstance(c, Tensor), f"MATMUL forward error: OP didn't return Tensor object, instead got {type(b)}"
        assert c.shape == (1, 1), f"MATMUL forward shape error: expected (1, 1), got {c.shape}"
        # 1*3 + 2*4 = 3 + 8 = 11
        assert c.data[0][0] == 11.0, f"MATMUL forward compute error: expected 11.0, got {c.data}"

        # Identity Matrix Check
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        identity = Tensor([[1.0, 0.0], [0.0, 1.0]])
        c = a @ identity
        assert np.array_equal(c.data, a.data), f"MATMUL forward identity error: expected c == a instead got {c.data} != {a.data}"

    def test_matmul_backward(self):
        x = Tensor(np.ones((2, 3)), requires_grad=True)
        w = Tensor(np.ones((3, 2)), requires_grad=True)
        out = x @ w
        out.backward()

        assert isinstance(x.grad, Tensor), f"MATMUL backward error: didn't compute gradient of *x*({type(x.grad)})"
        assert isinstance(w.grad, Tensor), f"MATMUL backward error: didn't compute gradient of *w*({type(w.grad)})"
        assert x.grad.data.shape == (2, 3), f"MATMUL backward X shape error: got {x.grad.shape}"
        assert w.grad.shape == (3, 2), f"MATMUL backward W shape error: got {w.grad.shape}"

        # dL/dx = grad_out @ w.T -> Ones(2,2) @ Ones(2,3) = Each elem is sum(col) = 2.0
        # dL/dw = x.T @ grad_out -> Ones(3,2) @ Ones(2,2) = Each elem is sum(row) = 2.0
        assert np.all(x.grad.data == 2.0), f"MATMUL backward value error (x)"
        assert np.all(w.grad.data == 2.0), f"MATMUL backward value error (w)"

    def test_transpose_forward(self):
        a = Tensor([[1.0, 2.0, 3.0]])
        b = a.T
        assert b.shape == (3, 1), f"TRANSPOSE forward shape error: expected (3, 1), got {b.shape}"
        assert b.data[0][0] == 1.0 and b.data[2][0] == 3.0

        c = a.transpose()
        assert c.shape == (3, 1), f"TRANSPOSE forward shape error: expected (3, 1), got {c.shape}"

    def test_transpose_backward(self):
        # y = x.T
        # if dy/dy = 1, then dy/dx = 1 (just transposed back)
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = a.T
        b.backward()

        # Gradient should be all ones, matching shape of a (2, 2)
        assert isinstance(a.grad, Tensor), f"TRANSPOSE backward error: didn't compute gradient of *a*({type(a.grad)})"
        assert a.grad.shape == (2, 2), f"TRANSPOSE backward shape error: expected (2, 2), got {a.shape}"
        assert np.all(a.grad.data == 1.0), f"TRANSPOSE backward error: wrongly computed gradients, expected all(1.0) got {a.grad.data}"

    def test_reshape_forward(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = a.reshape((4,))
        assert isinstance(b, Tensor), f"RESHAPE forward error: reshape didn't return Tensor object, instead got {type(b)}"
        assert b.shape == (4,), f"RESHAPE forward error: reshape expected (4,) instead got {b.shape}"
        assert np.array_equal(b.data, [1, 2, 3, 4]), f"RESHAPE forward error: data was mishandled when reshaping, expected [1, 2, 3, 4] and got {b.data}"

    def test_reshape_backward(self):
        a = Tensor(np.ones((2, 3)), requires_grad=True)
        b = a.reshape((6,))
        b.sum().backward() # Sum to create scalar loss
        assert isinstance(a.grad, Tensor), f"RESHAPE backward error: didn't compute gradient of *a*({type(a.grad)})"
        assert a.grad.shape == (2, 3), f"RESHAPE backward shape error: expected (2, 3) got {a.grad.shape}"
        assert np.all(a.grad.data == 1.0), f"RESHAPE backward error: Gradient compute error, expected all(1.0) got ({a.grad.data})"

    def test_relu_forward(self):
        a = Tensor([-1.0, 0.0, 1.0])
        b = a.relu()
        assert isinstance(b, Tensor), f"RELU forward erorr: didn't return a Tensor object, instead got {type(b)}"
        assert np.array_equal(b.data, [0.0, 0.0, 1.0]), f"RELU forward error: got {b.data}"

        a = Tensor([-5.0, -2.0])
        b = a.relu()
        assert isinstance(b, Tensor), f"RELU forward error: didn't return a Tensor object, instead got {type(b)}"
        assert np.array_equal(b.data, [0.0, 0.0]), f"RELU forward compute error: expected all(0.0) got {b.data}"

    def test_relu_backward(self):
        a = Tensor([-1.0, 2.0], requires_grad=True)
        b = a.relu()
        assert isinstance(b, Tensor), f"RELU backward error: didn't return a Tensor object instead got {type(b)}"
        b.backward()
        assert isinstance(a.grad, Tensor), f"RELU backward error: didn't compute gradient of *a*({type(a.grad)})"
        assert np.array_equal(a.grad.data, [0.0, 1.0]), f"RELU backward error: got {a.grad.data}"

    def test_leakyrelu_forward(self):
        a = Tensor([-1.0, 2.0])
        b = a.leaky_relu(slope=0.1)
        assert isinstance(b, Tensor), f"LEAKYRELU forward error: OP didn't return a Tensor object instead got {type(b)}"

        # -1 * 0.1 = -0.1, 2 stays 2
        assert np.allclose(b.data, [-0.1, 2.0]), f"LEAKYRELU forward error: expected [-0.1, 2.0] instead got {b.data}"

    def test_leakyrelu_backward(self):
        a = Tensor([-1.0, 2.0], requires_grad=True)
        b = a.leaky_relu(slope=0.1)
        b.backward()

        # grad of -1 is 0.1, grad of 2 is 1
        assert isinstance(a.grad, Tensor), f"LEAKYRELU backward error: didn't compute gradient of *a*({type(a.grad)})"
        assert np.allclose(a.grad.data, [0.1, 1.0]), f"LEAKYRELU backward error: got {a.grad.data}"

    def test_sum_forward(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = a.sum()
        assert isinstance(b, Tensor)
        assert b.data == 6.0, f"SUM forward error: got {b.data}"

        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = a.sum(axis=0)
        assert np.array_equal(b.data, [4.0, 6.0]), f"SUM axis=0 forward error: expected [4.0, 6.0] got {b.data}"

        c = a.sum(axis=1)
        assert np.array_equal(c.data, [3.0, 7.0]), f"SUM axis=1 forward error: expected [3.0, 7.0] got {c.data}"

    def test_sum_backward(self):
        # Simple backward sum
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = a.sum()
        b.backward()
        assert isinstance(a.grad, Tensor), f"SUM backward error: didn't compute gradient of *a*({type(a.grad)})"
        assert np.array_equal(a.grad.data, [1.0, 1.0, 1.0]), f"SUM backward grad compute error: expected all(1.0) got {a.grad.data}"

        # broadcasting logic
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.sum(axis=0)

        z = y * Tensor([2.0, 3.0])
        z = z.sum()
        z.backward()

        expected = [[2.0, 3.0], [2.0, 3.0]]
        assert isinstance(x.grad, Tensor), f"SUM backward error: didn't compute gradient of *x*({type(x.grad)})"
        assert np.array_equal(x.grad.data, expected), f"SUM axis backward error:\n{x.grad.data}"

    def test_exp_forward(self):
        a = Tensor(0.0)
        b = a.exp()
        assert isinstance(b, Tensor)
        assert b.data == 1.0, f"EXP forward error: e^0 != 1, got {b.data}"
    
    def test_exp_backward(self):
        a = Tensor(1.0, requires_grad=True)
        b = a.exp()
        b.backward()

        # d/dx(e^x) = e^x. at x=1, grad should be e
        assert isinstance(a.grad, Tensor), f"EXP backward error: didn't compute gradient of *a*({type(a.grad)})"
        assert abs(a.grad.data - np.exp(1.0)) < 1e-5, f"EXP backward error: got {a.grad.data}"

    def test_log_forward(self):
        a = Tensor(np.e)
        b = a.log()
        assert isinstance(b, Tensor)
        assert abs(b.data - 1.0) < 1e-5, f"LOG forward error: ln(e) != 1, got {b.data}"

    def test_log_backward(self):
        a = Tensor(2.0, requires_grad=True)
        b = a.log()
        b.backward()

        # d/dx(ln(x)) = 1/x. at x=2, grad should be 0.5
        assert isinstance(a.grad, Tensor), f"LOG backward error: didn't compute gradient of *a*({type(a.grad)})"
        assert a.grad.data == 0.5, f"LOG backward error: expected 0.5 got {a.grad.data}"

Test_Ops().test_all()
