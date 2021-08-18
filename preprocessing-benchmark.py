import timeit

s = """
import torch
import numpy as np
import numexpr as ne
from numba import jit

img = np.random.randint(low=0, high=255, size=(4096,4096,3), dtype=np.uint8)
img = np.random.randint(low=0, high=255, size=(1128,128,3), dtype=np.uint8)

def preprocess_input(x):
    return x.astype(np.float32, copy=False) / 127.5 - 1.

def preprocess_input_v2(x):
    return np.true_divide(x, 127.5, dtype=np.float32) - 1.

def preprocess_input_v3(x):
    return np.multiply(x, 1/127.5, dtype=np.float32) - 1.

def preprocess_input_v3_1(x):
    return np.multiply(x, np.array(1/127.5, dtype=np.float32), dtype=np.float32) - 1.

def preprocess_input_v3_2(x):
    return np.multiply(x, np.array(1/127.5, dtype=np.float32), dtype=np.float32) - np.array(1., dtype=np.float32)

def preprocess_input_v4(x):
    return np.subtract(np.multiply(x, np.array(1/127.5, dtype=np.float32), dtype=np.float32), np.array(1, dtype=np.float32), dtype=np.float32)

def albumentations_normalize(img, mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img

def preprocess_input_torch(x):
    x = torch.from_numpy(x).float()
    bias = torch.tensor(-1.0)
    x = bias.add(x, alpha=1/127.5)
    return x.numpy()

def preprocess_input_torch_v2(x):
    x = torch.from_numpy(x)
    bias = torch.tensor(-1).float()
    x = bias.add(x, alpha=1/127.5)
    return x.numpy()

def preprocess_input_numexpr(x):
    x = x.astype(np.float32, copy=False)
    return ne.evaluate("x / 127.5 - 1.")

def preprocess_input_numba(x):
    x = x.astype(np.float32, copy=False)
    return ff(x)

@jit(nopython=True, parallel = True)
def ff(x):
    return x / 127.5 - 1.

"""
f1 = "preprocess_input(img)"

f1_v2 = "preprocess_input_v2(img)"

f1_v3 = "preprocess_input_v3(img)"

f1_v3_1 = "preprocess_input_v3_1(img)"

f1_v3_2 = "preprocess_input_v3_2(img)"

f1_v4 = "preprocess_input_v4(img)"

f2 = "albumentations_normalize(img, 127.5, 127.5)"

f3 = "preprocess_input_torch(img)"

f3_v2 = "preprocess_input_torch_v2(img)"

f4 = "preprocess_input_numexpr(img)"

f5 = "preprocess_input_numba(img)"

for f in [f1, f1_v2, f1_v3, f1_v3_1, f1_v3_2, f1_v4, f2, f3, f3_v2, f4, f5]:
    print(f, timeit.timeit(setup = s, stmt = f, number = 1000))
