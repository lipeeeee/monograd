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
        # super().show_metrics()

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
        assert isinstance(b.grad, Tensor), f"Gradient type expected to be Tensor, got {type(a.grad)}"
        assert b.grad.data == 2.0, f"ADD backward error: expected 2.0, got {b.grad.data}"

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
        assert isinstance(b.grad, Tensor), f"Gradient type expected to be Tensor, got {type(a.grad)}"
        assert b.grad.data == 6.0, f"MUL backward error: expected 6.0, got {b.grad.data}"

Test_Ops().test_all()
