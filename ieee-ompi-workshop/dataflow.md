---
layout: page-ieee-ompi-workshop
title: 'Google Cloud Dataflow'
permalink: ieee-ompi-workshop/dataflow/
---

Table of contents:

- [Beam Programming](#beam-programming)
- [Enable Dataflow API](#enable-dataflow-api)
- [Building a Simple Data Transformation Pipeline](#building-a-simple-data-transformation-pipeline)
  - [Open Dataflow Dashboard](#open-dataflow-dashboard)
  - [Choose Dataflow Job](#choose-dataflow-job)
  - [Dataflow Transformation Pipeline](#dataflow-transformation-pipeline)
  - [Transformed Dataset in Bucket](#transformed-dataset-in-bucket)


Google Cloud Dataflow provides a serverless, parallel and distributed infrastructure for running jobs for batch and stream data processing.
One of the core strengths of Dataflow is its ability to almost seamlessly handle the switch from processing of batch historical data to streaming datasets while elegantly taking into consideration the perks of streaming processing such as windowing.
Dataflow is a major component for building an end-to-end ML production pipeline on GCP.

<a id="beam-programming"></a>

## Beam Programming
Apache Beam provides a set of broad concepts to simplify the process of building a transformation pipeline for distributed batch and stream jobs.

- **A Pipeline:** A Pipeline object wraps the entire operation and prescribes the transformation process by defining the input data source to the pipeline, how that data will be transformed and where the data will be written.
- **A PCollection:** A PCollection is used to define a data source. The data source can either be bounded or unbounded. A bounded data source referes to batch or historical data, whereas an unbounded data source refers to streaming data.
- **A PTransform:** PTransforms refers to a particular transformation task carried out on one or more PCollections in the pipeline. A number of core Beam transforms include:
  - ParDo: for parallel processing.
  - GroupByKey: for processing collections of key/value pairs.
  - CoGroupByKey: for a relational join of two or more key/value PCollections with the same key type.
  - Combine: for combining collections of elements or values in your data.
  - Flatten: for merging multiple PCollection objects.
  - Partition: splits a single PCollection into smaller collections. 
- **I/O Transforms:** These are PTransforms that read or write data to different external storage systems.

<div style="display: inline-block;width: 100%;">
<img src="/assets/ieee_ompi/dataflow-sequential-transform.png" alt="A Simple Linear Pipeline with Sequential Transforms." height="90%" width="90%" />
</div>

<a id="enable-dataflow-api"></a>

## Enable Dataflow API
(1). Go to API & Services Dashboard  
(2). Click `Enable API & services`  
(3). Search for `Dataflow API`  

<div style="display: inline-block;width: 100%;">
<img src="/assets/ieee_ompi/search-dataflow-api.png" alt="Search for Dataflow API." height="90%" width="90%" />
</div>

(4). Enable `Dataflow API`
<div style="display: inline-block;width: 100%;">
<img src="/assets/ieee_ompi/enable-dataflow-api.png" alt="Enable Dataflow API." height="90%" width="90%" />
</div>

<a id="building-a-simple-data-transformation-pipeline"></a>

## Building a Simple Data Transformation Pipeline
In this example, a transformation pipeline is built to pre-process the `crypto-markets.csv` dataset by removing the attributes that are not relevant for data modeling, including filtering only bitcoin crypto records.


```bash
%%bash
# create bucket
gsutil mb gs://ieee-ompi-datasets
```

    Creating gs://ieee-ompi-datasets/...



```bash
%%bash
# transfer data from Github to the bucket.
curl https://raw.githubusercontent.com/dvdbisong/IEEE-Carleton-and-OMPI-Machine-Learning-Workshop/master/data/crypto-markets/crypto-markets.csv | gsutil cp - gs://ieee-ompi-datasets/crypto-markets.csv
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0Copying from <STDIN>...
    100 47.0M  100 47.0M    0     0  25.6M      0  0:00:01  0:00:01 --:--:-- 25.6M
    / [1 files][    0.0 B/    0.0 B]                                                
    Operation completed over 1 objects.                                              



```python
# install the apache beam library and other important setup packages.
# restart the session after installing apache beam.
```


```bash
%%bash
source activate py2env
pip install google-cloud-dataflow
pip uninstall -y google-cloud-dataflow
conda install -y pytz==2018.4
pip install apache-beam[gcp]
```


```python
# import relevant libraries
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
```


```python
# transformation code
def run(project, source_bucket, target_bucket):
    import csv

    options = {
        'staging_location': 'gs://ieee-ompi-datasets/staging',
        'temp_location': 'gs://ieee-ompi-datasets/temp',
        'job_name': 'dataflow-crypto',
        'project': project,
        'max_num_workers': 24,
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'no_save_main_session': True,
        'runner': 'DataflowRunner'
      }
    options = beam.pipeline.PipelineOptions(flags=[], **options)
    
    crypto_dataset = 'gs://{}/crypto-markets.csv'.format(source_bucket)
    processed_ds = 'gs://{}/transformed-crypto-bitcoin'.format(target_bucket)

    pipeline = beam.Pipeline(options=options)

    # 0:slug, 3:date, 5:open, 6:high, 7:low, 8:close
    rows = (
        pipeline |
            'Read from bucket' >> ReadFromText(crypto_dataset) |
            'Tokenize as csv columns' >> beam.Map(lambda line: next(csv.reader([line]))) |
            'Select columns' >> beam.Map(lambda fields: (fields[0], fields[3], fields[5], fields[6], fields[7], fields[8])) |
            'Filter bitcoin rows' >> beam.Filter(lambda row: row[0] == 'bitcoin')
        )
        
    combined = (
        rows |
            'Write to bucket' >> beam.Map(lambda (slug, date, open, high, low, close): '{},{},{},{},{},{}'.format(
                slug, date, open, high, low, close)) |
            WriteToText(
                file_path_prefix=processed_ds,
                file_name_suffix=".csv", num_shards=2,
                shard_name_template="-SS-of-NN",
                header='slug, date, open, high, low, close')
        )

    pipeline.run()
```


```python
# execute transfomation
if __name__ == '__main__':
    print 'Run pipeline on the cloud'
    run(project='oceanic-sky-230504', source_bucket='ieee-ompi-datasets', target_bucket='ieee-ompi-datasets')
```

    Run pipeline on the cloud


### Open Dataflow Dashboard
<div style="display: inline-block;width: 100%;">
<img src="/assets/ieee_ompi/open-dataflow.png" alt="Open Dataflow." height="90%" width="90%" />
</div>

### Choose Dataflow Job
<div style="display: inline-block;width: 100%;">
<img src="/assets/ieee_ompi/choose-dataflow-job.png" alt="Choose Dataflow Job." height="90%" width="90%" />
</div>

### Dataflow Transformation Pipeline
<div style="display: inline-block;width: 100%;">
<img src="/assets/ieee_ompi/dataflow-transformation-pipeline.png" alt="Dataflow transformation pipeline." height="90%" width="90%" />
</div>

### Transformed Dataset in Bucket
<div style="display: inline-block;width: 100%;">
<img src="/assets/ieee_ompi/transformed-dataset.png" alt="Transformed dataset." height="90%" width="90%" />
</div>
