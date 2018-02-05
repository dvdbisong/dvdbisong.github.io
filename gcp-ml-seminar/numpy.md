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
Let's create a simple NumPy array:
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
> list_to_array = np.array(my_list)
> type(list_to_array)
'Output': numpy.ndarray
```

Let's explore other useful methods often employed for creating arrays
```python
# create an array from a range of numbers
> my_array = np.arange(10)
> print(m_array)
'Output': [0 1 2 3 4 5 6 7 8 9]
# create an array from start to end (exclusive) via a step size - (start, stop, step)
> my_array = np.arange(2, 10, 2)
'Output': [2 4 6 8]
```