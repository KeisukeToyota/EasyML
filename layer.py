import numpy as np


class BaseLayer(object):
  def __init__(self):
    self._params = None
    self._grads = None

  @property
  def params(self):
    return self._params

  @params.setter
  def params(self, x):
    if (np.ndarray is type(x)):
      self._params = x
    else:
      raise ValueError

  @property
  def grads(self):
    return self._grads

  @grads.setter
  def grads(self, x):
    if (np.ndarray is type(x)):
      self._grads = x
    else:
      raise ValueError

  @classmethod
  def forward(self, x: np.ndarray) -> np.ndarray:
    pass

  @classmethod
  def backward(self, dout: np.ndarray) -> np.ndarray:
    pass


class Sigmoid(BaseLayer):
  def __init__(self):
    super().__init__()
    self.params = np.array([])
    self.grads = np.array([])
    self.out = None

  def forward(self, x: np.ndarray) -> np.ndarray:
    out = 1 / (1 + np.exp(-x))
    self.out = out
    return out

  def backward(self, dout: np.ndarray) -> np.ndarray:
    dx = dout * (1.0 - self.out) * self.out
    return dx


class Affine(BaseLayer):
  def __init__(self, weight, bias):
    super().__init__(weight, bias)
