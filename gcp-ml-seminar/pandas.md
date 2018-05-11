---
layout: page-seminar
title: 'Pandas'
permalink: gcp-ml-seminar/pandas/
---

Table of contents:

- [Pandas Datastructures](#data_struct)
  - [Series](#series)
  - [DataFrames](#dataframes)
- [Data Indexing (Selection/ Subsets)](#data_indexing)
  - [Selecting a column from a DataFrame](#select_column)
  - [Selecting a row from a DataFrame](#select_row)
  - [Slice cells by row and column from a DataFrame](#slice)
- [DataFrame Manipulation](#dataframe_manipulation)
  - [Removing a Row/ Column](#removing_row)
  - [Adding a Row/ Column](#adding_row)
  - [Data Alignment](#data_alignment)
  - [Combining Datasets](#combining_datasets)
- [Handling Missing Data](#missing_data)
  - [Identifying missing data](#identify_missing)
  - [Removing missing data](#remove_missing_data)
  - [Imputing values into missing data](#imputing_values_missing_data)
- [Data Aggregation (Grouping)](#data_aggregation)
- [Statistical Summaries](#stat_summaries)
  - [Correlation](#correlation)
  - [Skewness](#skewness)
- [Importing Data](#importing_data)


Pandas is a specialized Python library for data analysis, especially on humongous datasets. It boasts easy to use functionality for reading and writing data, dealing with missing data, reshaping the dataset, massaging the data by slicing, indexing, inserting and deleting data variables and records. Pandas also have an important `groupBy` functionality for aggregating data for defined conditions - useful for plotting and computing data summaries for exploration.

Another key strength of Pandas is in re-ordering and cleaning time series data for time-series analysis. In short, Pandas is the go-to tool for data cleaning and data exploration.

<a name='data_struct'></a>

### Pandas Datastructures
Just like NumPy, Pandas can store and manipulate a multi-dimensional array of data. To handle this, Pandas has the `Series` and `DataFrame` data structures.

<a name='series'></a>

#### Series
The `Series` data structure is for storing a 1-Dimensional array (or vector) of data elements. A series data structure also provides labels to the data items in the form of an `index`. The user can specify this labels via the `index` parameter in the `Series` function, but if the `index` parameter is left unspecified, a default label of 0 to one minus the size of the data elements are assigned.

To begin with Pandas, we’ll start by importing the Pandas module:
```python
import pandas as pd
```

Let us consider an example of creating a `Series` data structure.
```python
# create a Series object
> my_series = pd.Series([2,4,6,8], index=['e1','e2','e3','e4'])
# print out data in Series data structure
> my_series
'Output': 
e1    2
e2    4
e3    6
e4    8
dtype: int64
# check the data type of the variable
> type(my_series)
'Output': pandas.core.series.Series
# return the elements of the Series data structure
> my_series.values
'Output': array([2, 4, 6, 8])
# retrieve elements from Series data structure based on their assigned indices
> my_series['e1']
'Output': 2
# return all indices of the Series data structure
> my_series.index
'Output': Index(['e1', 'e2', 'e3', 'e4'], dtype='object')
```

Elements in a Series data structure can be assigned the same indices
```python
# create a Series object with elements sharing indices
> my_series = pd.Series([2,4,6,8], index=['e1','e2','e1','e2'])
# note the same index assigned to various elements
> my_series
'Output': 
e1    2
e2    4
e1    6
e2    8
dtype: int64
# get elements using their index
> my_series['e1']
'Output': 
e1    2
e1    6
dtype: int64
```

<a name='dataframes'></a>

#### DataFrames
A DataFrame is a Pandas data structure for storing and manipulating 2-Dimensional arrays. A 2-Dimensional array is a table-like structure that is similar to an Excel spreadsheet or a relational database table. A DataFrame is a very natural form for storing structured datasets.

A DataFrame consists of rows and columns for storing records of information (in rows) across heterogeneous variables (in columns).

Let's see examples of working with DataFrames.
```python
# create a data frame
> my_DF = pd.DataFrame({'age': [15,17,21,29,25], \
            'state_of_origin':['Lagos', 'Cross River', 'Kano', 'Abia', 'Benue']})
> my_DF
'Output': 
   age state_of_origin
0   15           Lagos
1   17     Cross River
2   21            Kano
3   29            Abia
4   25           Benue
```

We will observe from the above example that a DataFrame is constructed from a dictionary of records where each value is a `Series` data structure. Also note that each row has an `index` that can be assigned when creating the DataFrame, else the default from 0 to one off the number of records in the DataFrame is used. Creating an index manually is usually not feasible except when working with small dummy datasets.

NumPy is frequently used together with Pandas. Let's import the NumPy library and use some of its functions to demonstrate other ways of creating a quick DataFrame.
```python
import numpy as np

# create a 3x3 dataframe of numbers from the normal distribution
> my_DF = pd.DataFrame(np.random.randn(3,3),\
            columns=['First','Second','Third'])
> my_DF
'Output': 
      First    Second     Third
0 -0.211218 -0.499870 -0.609792
1 -0.295363  0.388722  0.316661
2  1.397300 -0.894861  1.127306
# check the dimensions
> my_DF.shape
'Output': (3, 3)
```

Let's examine some other operations with DataFrames.
```python
# create a python dictionary
> my_dict = {'State':['Adamawa', 'Akwa-Ibom', 'Yobe', 'Rivers', 'Taraba'], \
            'Capital':['Yola','Uyo','Damaturu','Port-Harcourt','Jalingo'], \
            'Population':[3178950, 5450758, 2321339, 5198716, 2294800]}
> my_dict
'Output': 
{'Capital': ['Yola', 'Uyo', 'Damaturu', 'Port-Harcourt', 'Jalingo'],
 'Population': [3178950, 5450758, 2321339, 5198716, 2294800],
 'State': ['Adamawa', 'Akwa-Ibom', 'Yobe', 'Rivers', 'Taraba']}
# confirm dictionary type
> type(my_dict)
'Output': dict
# create DataFrame from dictionary
> my_DF = pd.DataFrame(my_dict)
> my_DF
'Output': 
         Capital  Population      State
0           Yola     3178950    Adamawa
1            Uyo     5450758  Akwa-Ibom
2       Damaturu     2321339       Yobe
3  Port-Harcourt     5198716     Rivers
4        Jalingo     2294800     Taraba
# check DataFrame type
> type(my_DF)
'Output': pandas.core.frame.DataFrame
# retrieve column names of the DataFrame
> my_DF.columns
'Output': Index(['Capital', 'Population', 'State'], dtype='object')
# the data type of `DF.columns` method is an Index
> type(my_DF.columns)
'Output': pandas.core.indexes.base.Index
# retrieve the DataFrame values as a NumPy ndarray
> my_DF.values
'Output': 
array([['Yola', 3178950, 'Adamawa'],
       ['Uyo', 5450758, 'Akwa-Ibom'],
       ['Damaturu', 2321339, 'Yobe'],
       ['Port-Harcourt', 5198716, 'Rivers'],
       ['Jalingo', 2294800, 'Taraba']], dtype=object)
# the data type of  `DF.values` method is an numpy ndarray
type(my_DF.values)
'Output': numpy.ndarray
```

In summary, a DataFrame is a tabular structure for storing a structured dataset where each column contains a `Series` data structure of records. Here's an illustration.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/pandas-DataStructure.png">
    <div class="figcaption" style="text-align: center;">
        Figure 1: Pandas Data structure
    </div>
</div>

Let's check the data type of each column in the DataFrame
```python
> my_DF.dtypes
'Output': 
Capital       object
Population     int64
State         object
dtype: object
```
An `object` data type in Pandas represents `Strings`

<a name='data_indexing'></a>

### Data Indexing (Selection/ Subsets)
Similar to NumPy, Pandas objects can index or subset the dataset to retrieve a specific sub-record of the larger dataset. Note that data indexing returns a new `DataFrame` or `Series` if a 2-D or 1-D array is retrieved. They do not, however, alter the original dataset. Let's go through some examples of indexing a Pandas DataFrame.

First let's create a dataframe. Observe the default integer indices assigned.
```python
# create the dataframe
> my_DF = pd.DataFrame({'age': [15,17,21,29,25], \
            'state_of_origin':['Lagos', 'Cross River', 'Kano', 'Abia', 'Benue']})
> my_DF
'Output':  
   age state_of_origin
0   15           Lagos
1   17     Cross River
2   21            Kano
3   29            Abia
4   25           Benue
```

<a name='select_column'></a>

#### Selecting a column from a DataFrame
Remember that the data type of a DataFrame column is a `Series` because it is a vector or 1-Dimensional array.
```python
> my_DF['age']
'Output':  
0    15
1    17
2    21
3    29
4    25
Name: age, dtype: int64
# check data type
> type(my_DF['age'])
'Output':  pandas.core.series.Series
```

<a name='select_column'></a>

#### Selecting a row from a DataFrame
Pandas makes use of two unique wrapper attributes for indexing rows from a `DataFrame` or a cell from a `Series` data structure. These attributes are the `iloc` and `loc` - they are also known as indexers. The `iloc` attribute allows you to select or slice row(s) of a DataFrame using the intrinsic Python index format whereas the `loc` attribute uses the explicit indices assigned to the DataFrame. If no explicit index is found, `loc` returns the same value as `iloc`.

Remember that the data type of a DataFrame row is a `Series` because it is a vector or 1-Dimensional array.

Let's select the first row from the DataFrame
```python
# using explicit indexing
> my_DF.loc[0]
'Output':  
age                   15
state_of_origin    Lagos
Name: 0, dtype: object
# using implicit indexing
> my_DF.iloc[0]
'Output':  
age                   15
state_of_origin    Lagos
Name: 0, dtype: object
# let's see the data type
> type(my_DF.loc[0])
'Output':  pandas.core.series.Series
```

Now let's create a DataFrame with explicit indexing and test out the `iloc` and `loc` methods. As you will see Pandas will return an error if we try to use `iloc` for explicit indexing or try to use `loc` for implicit Python indexing.

```python
> my_DF = pd.DataFrame({'age': [15,17,21,29,25], \
            'state_of_origin':['Lagos', 'Cross River', 'Kano', 'Abia', 'Benue']},\
            index=['a','a','b','b','c'])
# observe the string indices
> my_DF
'Output':  
   age state_of_origin
a   15           Lagos
a   17     Cross River
b   21            Kano
b   29            Abia
c   25           Benue
# select using explicit indexing
> my_DF.loc['a']
Out[196]: 
   age state_of_origin
a   15           Lagos
a   17     Cross River
# lets try to use loc for implicit indexing
> my_DF.loc[0]
'Output':  
    Traceback (most recent call last):
    TypeError: cannot do label indexing on <class 'pandas.core.indexes.base.Index'> 
        with these indexers [0] of <class 'int'>
```

<a name='slice'></a>

#### Slice cells by row and column from a DataFrame
First let's create a DataFrame. Remember, we use `iloc` when no explicit index or row labels are assigned.
```python
> my_DF = pd.DataFrame({'age': [15,17,21,29,25], \
            'state_of_origin':['Lagos', 'Cross River', 'Kano', 'Abia', 'Benue']})
> my_DF
'Output':  
   age state_of_origin
0   15           Lagos
1   17     Cross River
2   21            Kano
3   29            Abia
4   25           Benue
# select the third row and second column
> my_DF.iloc[2,1]
'Output': 'Kano'
# slice the first 2 rows - indexed from zero, excluding the final index
> my_DF.iloc[:2,]
'Output':  
   age state_of_origin
0   15           Lagos
1   17     Cross River
# slice the last three rows from the last column
> my_DF.iloc[-3:,-1]
'Output':  
2     Kano
3     Abia
4    Benue
Name: state_of_origin, dtype: object
```

<a name='dataframe_manipulation'></a>

### DataFrame Manipulation

<a name='removing_row'></a>

#### Removing a Row/ Column
In many cases during the data cleaning process, they may be a need to drop unwanted rows or data variables (i.e., columns). We typically do this using the `drop` function. The `drop` function has a parameter `axis` whose default is `0`. If `axis` is set to 1, it drops columns in a dataset, but if left at the default, rows are dropped from the dataset. 

Note that when a column or row is dropped a new `DataFrame` or `Series` is returned without altering the original data structure. Let's see some examples.
```python
# our data frame
> my_DF = pd.DataFrame({'age': [15,17,21,29,25], \
            'state_of_origin':['Lagos', 'Cross River', 'Kano', 'Abia', 'Benue']})
> my_DF
'Output':  
   age state_of_origin
0   15           Lagos
1   17     Cross River
2   21            Kano
3   29            Abia
4   25           Benue
# drop the 3rd and 4th column
> my_DF.drop([2,4])
'Output':  
   age state_of_origin
0   15           Lagos
1   17     Cross River
3   29            Abia
# drop the `age` column
> my_DF.drop('age', axis=1)
'Output':  
  state_of_origin
0           Lagos
1     Cross River
2            Kano
3            Abia
4           Benue
```

<a name='adding_row'></a>

#### Adding a Row/ Column
We can add a new column to a Pandas DataFrame by using the `assign` method
```python
# show dataframe
> my_DF = pd.DataFrame({'age': [15,17,21,29,25], \
            'state_of_origin':['Lagos', 'Cross River', 'Kano', 'Abia', 'Benue']})
> my_DF
'Output':  
   age state_of_origin
0   15           Lagos
1   17     Cross River
2   21            Kano
3   29            Abia
4   25           Benue
# add column to data frame
> my_DF = my_DF.assign(capital_city = pd.Series(['Ikeja', 'Calabar', \
                                                'Kano', 'Umuahia', 'Makurdi']))
> my_DF
'Output':  
   age state_of_origin capital_city
0   15           Lagos        Ikeja
1   17     Cross River      Calabar
2   21            Kano         Kano
3   29            Abia      Umuahia
4   25           Benue      Makurdi
```

We can also add a new DataFrame column by computing some function on another column. Let's take an example by adding a column computing the absolute difference of the ages from their mean.
```python
> mean_of_age = my_DF['age'].mean()
> my_DF['diff_age'] = my_DF['age'].map(lambda x: abs(x-mean_of_age))
> my_DF
'Output':  
   age state_of_origin  diff_age
0   15           Lagos       6.4
1   17     Cross River       4.4
2   21            Kano       0.4
3   29            Abia       7.6
4   25           Benue       3.6
```

Typically in practice, a fully formed dataset is converted into Pandas for cleaning and data analysis, which does not ideally involve adding a new observation to the dataset. But in the event that this is desired, we can use the `append()` method to achieve this. However, it may not be a computationally efficient action. Let's see an example.

```python
# show dataframe
> my_DF = pd.DataFrame({'age': [15,17,21,29,25], \
            'state_of_origin':['Lagos', 'Cross River', 'Kano', 'Abia', 'Benue']})
> my_DF
'Output':  
   age state_of_origin
0   15           Lagos
1   17     Cross River
2   21            Kano
3   29            Abia
4   25           Benue
# add a row to data frame
> my_DF = my_DF.append(pd.Series([30 , 'Osun'], index=my_DF.columns), \
                                                        ignore_index=True)
> my_DF
'Output':  
   age state_of_origin
0   15           Lagos
1   17     Cross River
2   21            Kano
3   29            Abia
4   25           Benue
5   30            Osun
```
We observe that adding a new row involves passing to the `append` method, a `Series` object with the `index` attribute set to the columns of the main DataFrame. Since typically, in given datasets, the index is nothing more than the assigned defaults, we set the attribute `ignore_index` to create a new set of default index values with the new row(s).

<a name='data_alignment'></a>

#### Data Alignment
Pandas utilizes data alignment to align indices when performing some binary arithmetic operation on DataFrames. If two or more DataFrames in an arithmetic operation do not share a common index, a `NaN` is introduced denoting missing data. Let's see examples of this.

```python
# create a 3x3 dataframe - remember randint(low, high, size)
> df_A = pd.DataFrame(np.random.randint(1,10,[3,3]),\
            columns=['First','Second','Third'])
> df_A
'Output':  
   First  Second  Third
0      2       3      9
1      8       7      7
2      8       6      4
# create a 4x3 dataframe
> df_B = pd.DataFrame(np.random.randint(1,10,[4,3]),\
            columns=['First','Second','Third'])
> df_B
'Output':  
   First  Second  Third
0      3       6      3
1      2       2      1
2      9       3      8
3      2       9      2
# add df_A and df_B together
> df_A + df_B
'Output':  
   First  Second  Third
0    5.0     9.0   12.0
1   10.0     9.0    8.0
2   17.0     9.0   12.0
3    NaN     NaN    NaN
> divide both dataframes
> df_A / df_B
'Output':  
      First  Second  Third
0  0.666667     0.5    3.0
1  4.000000     3.5    7.0
2  0.888889     2.0    0.5
3       NaN     NaN    NaN
```

If we do not want a `NaN` signifying missing values to be imputed, we can use the `fill_value` attribute to substitute with a default value. However, to take advantage of the `fill_value` attribute, we have to use the Pandas arithmetic methods: `add()`, `sub()`, `mul()`, `div()`, `floordiv()`, `mod()`, and `pow()` for addition, subtraction, multiplication, integer division, numeric division, reminder division and exponentiation. Let's see examples.

```python
> df_A.add(df_B, fill_value=10)
'Output':  
   First  Second  Third
0    5.0     9.0   12.0
1   10.0     9.0    8.0
2   17.0     9.0   12.0
3   12.0    19.0   12.0
```

<a name='combining_datasets'></a>

#### Combining Datasets
We may need to combine two or more data sets together, Pandas provides methods for such operations. We would consider the simple case of combining data frames with shared column names using the `concat` method.

```python
# combine two dataframes column-wise
> pd.concat([df_A, df_B])
'Output':  
   First  Second  Third
0      2       3      9
1      8       7      7
2      8       6      4
0      3       6      3
1      2       2      1
2      9       3      8
3      2       9      2
```

Observe that the `concat` method preserves indices by default. We can also concatenate or combine two dataframes by rows (or horizontally). This is done by setting the `axis` parameter to `1`.
```python
# combine two dataframes horizontally
> pd.concat([df_A, df_B], axis=1)
'Output':  
Out[246]: 
   First  Second  Third  First  Second  Third
0    2.0     3.0    9.0      3       6      3
1    8.0     7.0    7.0      2       2      1
2    8.0     6.0    4.0      9       3      8
3    NaN     NaN    NaN      2       9      2
```

<a name='missing_data'></a>

### Handling Missing Data
Dealing with missing data is an integral part of the Data cleaning/ data analysis process. Moreover, some machine learning algorithms will not work in the presence of missing data. Let's see some simple Pandas methods for identifying and removing missing data, as well as imputing values into missing data.

<a name='identify_missing'></a>

#### Identifying missing data
```python
# lets create a data frame with missing data
> my_DF = pd.DataFrame({'age': [15,17,np.nan,29,25], \
            'state_of_origin':['Lagos', 'Cross River', 'Kano', 'Abia', np.nan]})
> my_DF
'Output':  
    age state_of_origin
0  15.0           Lagos
1  17.0     Cross River
2   NaN            Kano
3  29.0            Abia
4  25.0             NaN
```
Let's check for missing data in this data frame. The `isnull()` method will return `True` where there is a missing data, whereas the `notnull()` function returns `False`.
```python
> my_DF.isnull()
'Output':  
     age  state_of_origin
0  False            False
1  False            False
2   True            False
3  False            False
4  False             True
```

However, if we want a single answer (i.e., either `True` or `False`) to report if there is a missing data in the data frame, we will first convert the DataFrame to a NumPy array and use the function `any()`.

The `any` function returns `True` when at least one of the elements in the dataset is `True`. In this case, `isnull()` returns a DataFrame of booleans where `True` designates a cell with a missing value.

Let's see how that works
```python
> my_DF.isnull().values.any()
'Output':  True
```

<a name='remove_missing_data'></a>

#### Removing missing data
Pandas has a function `dropna()` which is used to filter or remove missing data from a DataFrame. `dropna()` returns a new DataFrame without missing data. Let's see examples of how this works

```python
# let's see our dataframe with missing data
> my_DF = pd.DataFrame({'age': [15,17,np.nan,29,25], \
            'state_of_origin':['Lagos', 'Cross River', 'Kano', 'Abia', np.nan]})
> my_DF
'Output':  
    age state_of_origin
0  15.0           Lagos
1  17.0     Cross River
2   NaN            Kano
3  29.0            Abia
4  25.0             NaN
# let's run dropna() to remove all rows with missing values
> my_DF.dropna()
'Output':  
    age state_of_origin
0  15.0           Lagos
1  17.0     Cross River
3  29.0            Abia
```
As we will observe from the above code-block, `dropna()` drops all rows that contain a missing value. But we may not want that. We may rather, for example, want to drop columns with missing data, or drop rows where all the observations are missing or better still remove consequent on the number of observations present in a particular row.

Let's see examples of this options. First let's expand our example dataset
```python
> my_DF = pd.DataFrame({'Capital': ['Yola', np.nan, np.nan, 'Port-Harcourt', 'Jalingo'],
 'Population': [3178950, np.nan, 2321339, np.nan, 2294800],
 'State': ['Adamawa', np.nan, 'Yobe', np.nan, 'Taraba'],
 'LGAs': [22, np.nan, 17, 23, 16]})
> my_DF
'Output':  
         Capital  LGAs  Population    State
0           Yola  22.0   3178950.0  Adamawa
1            NaN   NaN         NaN      NaN
2            NaN  17.0   2321339.0     Yobe
3  Port-Harcourt  23.0         NaN      NaN
4        Jalingo  16.0   2294800.0   Taraba
```

Drop columns with `NaN`. This option is not often used in practice.
```python
> my_DF.dropna(axis=1)
'Output':  
Empty DataFrame
Columns: []
Index: [0, 1, 2, 3, 4]
```

Drop rows where all the observations are missing.
```python
> my_DF.dropna(how='all')
'Output':  
         Capital  LGAs  Population    State
0           Yola  22.0   3178950.0  Adamawa
2            NaN  17.0   2321339.0     Yobe
3  Port-Harcourt  23.0         NaN      NaN
4        Jalingo  16.0   2294800.0   Taraba
```

Drop rows based on an observation threshold. By adjusting the `thresh` attribute, we can drop rows where the number of observations in the row is less than the `thresh` value.
```python
# drop rows where number of NaN are less than 3
> my_DF.dropna(thresh=3)
'Output':  
   Capital  LGAs  Population    State
0     Yola  22.0   3178950.0  Adamawa
2      NaN  17.0   2321339.0     Yobe
4  Jalingo  16.0   2294800.0   Taraba
```

<a name='imputing_values_missing_data'></a>

#### Imputing values into missing data
Imputing values as substitutes for missing data is a standard practice in preparing data for Machine Learning. Pandas has a `fillna()` function for this purpose. A simple approach is to fill `NaN`s with zeros.
```python
> my_DF.fillna(0) # we can also run my_DF.replace(np.nan, 0)
'Output':  
         Capital  LGAs  Population    State
0           Yola  22.0   3178950.0  Adamawa
1              0   0.0         0.0        0
2              0  17.0   2321339.0     Yobe
3  Port-Harcourt  23.0         0.0        0
4        Jalingo  16.0   2294800.0   Taraba
```

Another tactic is to fill missing values with the mean of the column value.
```python
> my_DF.fillna(my_DF.mean())
'Output':
         Capital  LGAs  Population    State
0           Yola  22.0   3178950.0  Adamawa
1            NaN  19.5   2598363.0      NaN
2            NaN  17.0   2321339.0     Yobe
3  Port-Harcourt  23.0   2598363.0      NaN
4        Jalingo  16.0   2294800.0   Taraba
```

<a name='data_aggregation'></a>

### Data Aggregation (Grouping)
We will touch briefly on a common practice in Data Science, and that is grouping a set of data attributes, either for retrieving some group statistics or applying a particular set of functions to the group. Grouping is commonly used for data exploration and plotting graphs to understand more about the data set. Missing data are automatically excluded in a grouping operation.

Let's see examples of how this works
```python
# create a data frame
> my_DF = pd.DataFrame({'Sex': ['M', 'F', 'M', 'F','M', 'F','M', 'F'],
 'Age': np.random.randint(15,60,8),
 'Salary': np.random.rand(8)*10000})
> my_DF
'Output':
   Age       Salary Sex
0   54  6092.596170   M
1   57  3148.886141   F
2   37  5960.916038   M
3   23  6713.133849   F
4   34  5208.240349   M
5   25  2469.118934   F
6   50  1277.511182   M
7   54  3529.201109   F
```

Let's find the mean age and salary for observations in our dataset grouped by `Sex`.
```python
> my_DF.groupby('Sex').mean()
'Output':
       Age       Salary
Sex                    
F    39.75  3965.085008
M    43.75  4634.815935
```

We can group by more than one variable. In this case for each Sex group, also group the age and find the mean of the other numeric variables.
```python
> my_DF.groupby([my_DF['Sex'], my_DF['Age']]).mean()
'Output':
              Salary
Sex Age             
F   23   6713.133849
    25   2469.118934
    54   3529.201109
    57   3148.886141
M   34   5208.240349
    37   5960.916038
    50   1277.511182
    54   6092.596170
```

Also, we can use a variable as a group key to run a group function on another variable or sets of variables.
```python
> my_DF['Age'].groupby(my_DF['Salary']).mean()
'Output':
Salary
1277.511182    50
2469.118934    25
3148.886141    57
3529.201109    54
5208.240349    34
5960.916038    37
6092.596170    54
6713.133849    23
Name: Age, dtype: int64
```

<a name='stat_summaries'></a>

### Statistical Summaries
Descriptive statistics is an essential component of the Data Science pipeline. By investigating the properties of the dataset, we can gain a better understanding of the data and the relationship between the variables. This information is useful in making decisions about the type of data transformations to carry out or the types of learning algorithms to spot-check. Let's see some examples of simple statistical functions in Pandas.

First, we'll create a Pandas dataframe
```python
> my_DF = pd.DataFrame(np.random.randint(10,80,[7,4]),\
            columns=['First','Second','Third', 'Fourth'])
'Output':
   First  Second  Third  Fourth
0     47      32     66      52
1     37      66     16      22
2     24      16     63      36
3     70      47     62      12
4     74      61     44      18
5     65      73     21      37
6     44      47     23      13
```

Use the `describe` function to obtain summary statistics of a dataset. Eight statistical measures are displayed. They are: count, mean, standard deviation, minimum value, 25th percentile, 50th percentile or median, 75th percentile and the maximum value.
```python
> my_DF.describe()
'Output':
           First     Second      Third     Fourth
count   7.000000   7.000000   7.000000   7.000000
mean   51.571429  48.857143  42.142857  27.142857
std    18.590832  19.978560  21.980511  14.904458
min    24.000000  16.000000  16.000000  12.000000
25%    40.500000  39.500000  22.000000  15.500000
50%    47.000000  47.000000  44.000000  22.000000
75%    67.500000  63.500000  62.500000  36.500000
max    74.000000  73.000000  66.000000  52.000000
```

<a name='correlation'></a>

#### Correlation
Correlation shows how much relationship exists between two variables. Parametric machine learning methods such as logistic and linear regression can take a performance hit when variables are highly correlated. The correlation values range from `-1` to `1`, with `0` indicating no correlation at all. `-1` signifies that the variables are strongly negatively correlated while `1` shows that the variables are strongly positively correlated. In practice, it is safe to eliminate variables that have a correlation value greater than `-0.7` or `0.7`. A common correlation estimate in use is the Pearson’s Correlation Coefficient.
```python
> my_DF.corr(method='pearson')
'Output':
           First    Second     Third    Fourth
First   1.000000  0.587645 -0.014100 -0.317333
Second  0.587645  1.000000 -0.768495 -0.345265
Third  -0.014100 -0.768495  1.000000  0.334169
Fourth -0.317333 -0.345265  0.334169  1.000000
```

<a name='skewness'></a>

#### Skewness
Another important statistical metric is the skewness of the dataset. Skewness is when a bell-shaped or Normal distribution is shifted towards the right or the left. Pandas offer a convenient function called `skew()` to check the skewness of each variable. Values close to `0` are more normally distributed with less skew.

```python
> my_DF.skew()
'Output':
First    -0.167782
Second   -0.566914
Third    -0.084490
Fourth    0.691332
dtype: float64
```

<a name='importing_data'></a>

### Importing Data
Again, getting data into the programming environment for analysis is a fundamental and first step for any data analytics or machine learning task. In practice, data usually comes in a comma separated value, `csv` format.

```python
> my_DF = pd.read_csv('link_to_file/csv_file', sep=',', header = None)
```

To export a DataFrame back to `csv`
```python
> my_DF.to_csv('file_name.csv')
```