---
layout: page-seminar
title: 'NumPy'
permalink: gcp-ml-seminar/numpy/
---

NumPy is a Python library optimized for numerical computing. It bears close semblance with MATLAB, and is equally as powerful when used in conjunction with other packages such as SciPy for various scientific functions, Matplotlib for visualization and Pandas for data management.

NumPy core strength lies in its ability to create and manipulate $$n$$-dimensional arrays. This is particularly critical for building Machine learning and Deep learning models. Data is often represented in a matrix-like grid of rows and columns, where each row represents an observation and each column a variable or feature. Hence, NumPy's 2-Dimensional arrays is a natural fit for storing and manipulating datasets.

This tutorial will cover the basics of NumPy to get you very comfortable working with the package and also get you to appreciate the thinking behind how NumPy works. This understanding forms a foundation from which one can extend and seek solutions from the NumPy reference documentation when a specific functionality is needed.

To begin using NumPy, we'll start by importing the NumPy module:
```python
import numpy as np
```

### NumPy Arrays
Let's create a simple 1-D NumPy array:
```python
> my_array = np.array([2,4,6,8,10])
> my_array
'Output': array([ 2,  4,  6,  8, 10])
# the data-type of a NumPy array is the ndarray
> type(my_array)
'Output': numpy.ndarray
# a NumPy 1-D array can also be seen a vector with 1 dimension
> my_array.ndim
'Output': 1
# check the shape to get the number of rows and columns in the array \
# read as (rows, columns)
> my_array.shape
'Output': (5,)
```

We can also create an array from a Python list
```python
> my_list = [9, 5, 2, 7]
> type(my_list)
'Output': list
# convert a list to a numpy array
> list_to_array = np.array(my_list) # or np.asarray(my_list)
> type(list_to_array)
'Output': numpy.ndarray
```

Let's explore other useful methods often employed for creating arrays
```python
# create an array from a range of numbers
> np.arange(10)
'Output': [0 1 2 3 4 5 6 7 8 9]
# create an array from start to end (exclusive) via a step size - (start, stop, step)
> np.arange(2, 10, 2)
'Output': [2 4 6 8]
# create a range of points between two numbers
> np.linspace(2, 10, 5)
'Output': array([  2.,   4.,   6.,   8.,  10.])
# create an array of ones
> np.ones(5)
'Output': array([ 1.,  1.,  1.,  1.,  1.])
# create an array of zeros
> np.zeros(5)
'Output': array([ 0.,  0.,  0.,  0.,  0.])
```

### NumPy Datatypes
NumPy boasts a broad range of numerical datatypes in comparison with vanilla Python. This extended datatype support is useful for dealing with different kinds of signed and unsigned integer and floating-point numbers and well as booleans and complex numbers for scientific computation. NumPy datatypes include the `bool_`, `int`(8,16,32,64), `uint`(8,16,32,64), `float`(16,32,64), `complex`(64,128) as well as the `int_`, `float_` and `complex_` to mention just a few.

The datatypes with a `_` appended are base Python datatypes converted to NumPy datatypes. The parameter `dtype` is used to assign a datatype to a NumPy function. The default NumPy type is `float_`. Also, Numpy infers contiguous arrays of the same type.

Let's explore a bit with NumPy datatypes:
```python
# ints
> my_ints = np.array([3, 7, 9, 11])
> my_ints.dtype
'Output': dtype('int64')

# floats
> my_floats = np.array([3., 7., 9., 11.])
> my_floats.dtype
'Output': dtype('float64')

# non-contiguous types - default: float
> my_array = np.array([3., 7., 9, 11])
> my_array.dtype
'Output': dtype('float64')

# manually assigning datatypes
> my_array = np.array([3, 7, 9, 11], dtype="float64")
> my_array.dtype
'Output': dtype('float64')
```

### Indexing + Fancy Indexing (1-D)
We can index a sigle element of a NumPy 1-D array similar to how we index a Python list.
```python
# create a random numpy 1-D array
> my_array = np.random.rand(10)
> my_array
'Output': array([ 0.7736445 ,  0.28671796,  0.61980802,  0.42110553,  0.86091567,
                  0.93953255,  0.300224  ,  0.56579416,  0.58890282,  0.97219289])
# index the first element
> my_array[0]
'Output': 0.77364449999999996
# index the last element
> my_array[-1]
'Output': 0.97219288999999998
```

Fancy Indexing in NumPy are advanced mechanisms for indexing array elements based on integers or boolean. This technique is also called <span style="color:green">*masking*</span>.

#### Boolean Mask
Let's index all the even integers in the array using a boolean mask.
```python
# create 10 random integers between 1 and 20
> my_array = np.random.randint(1, 20, 10)
> my_array
'Output': array([14,  9,  3, 19, 16,  1, 16,  5, 13,  3])
# index all even integers in the array using a boolean mask
> my_array[my_array % 2 == 0]
'Output': array([14, 16, 16])
```

Observe that the code `my_array % 2 == 0` output's an array of booleans
```python
> my_array % 2 == 0
'Output': array([ True, False, False, False,  True, False,  True, False, False, False], dtype=bool)
```

#### Integer Mask
Let's select all elements with **even indices** in the array.
```python
# create 10 random integers between 1 and 20
> my_array = np.array([14,  9,  3, 19, 16,  1, 16,  5, 13,  3])
> my_array
'Output': array([14,  9,  3, 19, 16,  1, 16,  5, 13,  3])
> my_array[np.arange(1,10,2)]
'Output': array([ 9, 19,  1,  5,  3])
```

Remember that array indices are indexed from `0`. So the second element, `9` is in the first index `1`.
```python
> np.arange(1,10,2)
'Output': array([1, 3, 5, 7, 9])
```

### Slicing a 1-D Array
Slicing a NumPy array is also similar to slicing a Python list.
```python
> my_array = np.array([14,  9,  3, 19, 16,  1, 16,  5, 13,  3])
> my_array
'Output': array([14,  9,  3, 19, 16,  1, 16,  5, 13,  3])
# slice the first 2 elements
> my_array[:2]
'Output': array([14,  9])
# slice the last 3 elements
> my_array[-3:]
'Output': array([ 5, 13,  3])
```

### Basic Math Operations on Arrays
The core power of NumPy is in its highly optimized vectorized functions for various mathematical, arithmetic and string operations. We'll explore a couple of basic arithmetic with NumPy 1-D arrays.
```python
# create an array of even numbers between 2 and 10
> my_array = np.arange(2,11,2)
'Output': array([ 2,  4,  6,  8, 10])
# sum of array elements
> np.sum(my_array) # or my_array.sum()
'Output': 30
# square root
> np.sqrt(my_array)
'Output': array([ 1.41421356,  2.        ,  2.44948974,  2.82842712,  3.16227766])
# log
> np.log(my_array)
'Output': array([ 0.69314718,  1.38629436,  1.79175947,  2.07944154,  2.30258509])
# exponent
> np.exp(my_array)
'Output': array([  7.38905610e+00,   5.45981500e+01,   4.03428793e+02,
                   2.98095799e+03,   2.20264658e+04])
```