Tzvi Aviv
November 9th, 2021
Explore the creation of the dataset for hospital mortality prediction from mimic3



```python
!pwd
```

    /ssd003/home/taviv/mimic3-benchmarks



```python
import pandas as pd
```


```python
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
```


```python
from mimic3models import common_utils
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader

from mimic3models.preprocessing import Discretizer, Normalizer

def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    # ret = common_utils.read_chunk(reader, 100)
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'])




```

parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default='/ssd003/home/taviv/mimic3-benchmarks/data/in-hospital-mortality/')
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='/ssd003/home/taviv/mimic3-benchmarks/data_ta')
args = parser.parse_args()


```python
! ls /ssd003/home/taviv/mimic3-benchmarks/data/in-hospital-mortality/
```

    test  test_listfile.csv  train	train_listfile.csv  val_listfile.csv



```python
# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir='/ssd003/home/taviv/mimic3-benchmarks/data/in-hospital-mortality/train',
                                         listfile='/ssd003/home/taviv/mimic3-benchmarks/data/in-hospital-mortality/train_listfile.csv',
                                         period_length=48.0)

```


```python
# Build readers, discretizers, normalizers
test_reader = InHospitalMortalityReader(dataset_dir='/ssd003/home/taviv/mimic3-benchmarks/data/in-hospital-mortality/test',
                                         listfile='/ssd003/home/taviv/mimic3-benchmarks/data/in-hospital-mortality/test_listfile.csv',
                                         period_length=48.0)
```


```python
print('Reading data and extracting features ...')
(train_X, train_y, train_names) = read_and_extract_features(train_reader, "all", "all")
```

    Reading data and extracting features ...



```python
print('Reading data and extracting features ...')
(test_X, test_y, test_names) = read_and_extract_features(test_reader, "all", "all")
```

    Reading data and extracting features ...



```python
test_X.shape
```




    (3236, 714)




```python
train_X.shape
```




    (14681, 714)




```python
train_names[0]
```




    '3977_episode4_timeseries.csv'




```python
#we want to get the subject id from train_names and add to the arrays or pd dfs
```


```python
train_df = pd.DataFrame(train_X)
train_df["label"]=train_y
```


```python
train_df["file"]=train_names
```


```python
train_df[['SUBJ_ID','episode', 'ts']] = train_df['file'].str.split('_',expand=True)
```


```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>709</th>
      <th>710</th>
      <th>711</th>
      <th>712</th>
      <th>713</th>
      <th>label</th>
      <th>file</th>
      <th>SUBJ_ID</th>
      <th>episode</th>
      <th>ts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3977_episode4_timeseries.csv</td>
      <td>3977</td>
      <td>episode4</td>
      <td>timeseries.csv</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>7.32</td>
      <td>7.280000</td>
      <td>0.032404</td>
      <td>-0.440867</td>
      <td>8.0</td>
      <td>0</td>
      <td>97271_episode1_timeseries.csv</td>
      <td>97271</td>
      <td>episode1</td>
      <td>timeseries.csv</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>7.40</td>
      <td>7.320000</td>
      <td>0.043205</td>
      <td>0.595170</td>
      <td>12.0</td>
      <td>0</td>
      <td>29742_episode1_timeseries.csv</td>
      <td>29742</td>
      <td>episode1</td>
      <td>timeseries.csv</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.00</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0</td>
      <td>3362_episode1_timeseries.csv</td>
      <td>3362</td>
      <td>episode1</td>
      <td>timeseries.csv</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>7.38</td>
      <td>7.376667</td>
      <td>0.004714</td>
      <td>-0.707107</td>
      <td>6.0</td>
      <td>0</td>
      <td>2187_episode2_timeseries.csv</td>
      <td>2187</td>
      <td>episode2</td>
      <td>timeseries.csv</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 719 columns</p>
</div>



"file", and "ts" columns can be removed


```python
train_df.SUBJ_ID.nunique()
```




    12565




```python
train_df.episode.value_counts().plot(kind='bar')
```




    <AxesSubplot:>




    
![png](output_21_1.png)
    



```python
#is mortality related to the episode number?
```


```python
train_grp = train_df[['label','episode']].groupby('episode')
```


```python
mr = pd.merge(train_grp.count(),train_grp.sum(), how = 'left', right_index=True, left_index=True)
```


```python
mr.columns=['n', 'died']
```


```python
mr['m_rate'] = mr.died/mr.n
```


```python
mr.sort_values('m_rate', ascending=False, inplace=True)
```


```python
mr.m_rate.plot(kind='bar')
```




    <AxesSubplot:xlabel='episode'>




    
![png](output_28_1.png)
    



```python
#high mortality rate in episode 7 (?)
```

## now lets see mortality rates in the test data


```python
test_df = pd.DataFrame(test_X)
test_df["label"]=test_y
test_df["file"]=test_names
test_df[['SUBJ_ID','episode', 'ts']] = test_df['file'].str.split('_',expand=True)
```


```python
test_grp = test_df[['label','episode']].groupby('episode')
mr_test = pd.merge(test_grp.count(),test_grp.sum(), how = 'left', right_index=True, left_index=True)
```


```python
mr_test.columns=['n', 'died']
mr_test['m_rate'] = mr_test.died/mr_test.n
mr_test.sort_values('m_rate', ascending=False, inplace=True)
```


```python
mr_test.m_rate.plot(kind='bar')
```




    <AxesSubplot:xlabel='episode'>




    
![png](output_34_1.png)
    



```python
#in the test set we get very high mortality rates in the 10th episode 
```


```python
test_df.SUBJ_ID.nunique()
```




    2763




```python
#another way to do it is from the listfile.csv
```


```python
files = pd.read_csv("data/in-hospital-mortality/train_listfile.csv")
```


```python
files.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stay</th>
      <th>y_true</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3977_episode4_timeseries.csv</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>97271_episode1_timeseries.csv</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29742_episode1_timeseries.csv</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3362_episode1_timeseries.csv</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2187_episode2_timeseries.csv</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
files[['SUBJ_ID','episode', 'ts']] = files['stay'].str.split('_',expand=True)
```


```python
files.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stay</th>
      <th>y_true</th>
      <th>SUBJ_ID</th>
      <th>episode</th>
      <th>ts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3977_episode4_timeseries.csv</td>
      <td>0</td>
      <td>3977</td>
      <td>episode4</td>
      <td>timeseries.csv</td>
    </tr>
    <tr>
      <th>1</th>
      <td>97271_episode1_timeseries.csv</td>
      <td>0</td>
      <td>97271</td>
      <td>episode1</td>
      <td>timeseries.csv</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29742_episode1_timeseries.csv</td>
      <td>0</td>
      <td>29742</td>
      <td>episode1</td>
      <td>timeseries.csv</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3362_episode1_timeseries.csv</td>
      <td>0</td>
      <td>3362</td>
      <td>episode1</td>
      <td>timeseries.csv</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2187_episode2_timeseries.csv</td>
      <td>0</td>
      <td>2187</td>
      <td>episode2</td>
      <td>timeseries.csv</td>
    </tr>
  </tbody>
</table>
</div>




```python
files.SUBJ_ID.nunique()
```




    12565




```python
files.shape
```




    (14681, 5)




```python

```
