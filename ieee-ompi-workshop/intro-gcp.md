---
layout: page-ieee-ompi-workshop
title: 'Introduction to Google Cloud Platform'
permalink: ieee-ompi-workshop/intro-gcp/
---

Table of contents:

- [The Computational Cost of Building ML Products](#the-computational-cost-of-building-ml-products)
- [Why GCP?](#why-gcp)
- [GCP Product and Service Offerings](#gcp-product-and-service-offerings)
  - [Cloud Compute](#cloud-compute)
  - [Cloud Storage](#cloud-storage)
  - [Big Data/ Analytics](#big-data-analytics)
  - [Cloud AI](#cloud-ai)


Google Cloud Platform (GCP) offers a wide range of services for securing, storing, serving and analyzing data. These cloud services form a secure cloud perimeter for data, where different operations and transformations can be carried out on the data without it ever leaving the cloud ecosystem.

GCP is a simple, yet powerful, and cost effective cloud option for building large-scale machine learning models. It boasts a rich set of products to simplify the process of performaing large-scale data analytics, model training and model deployment for inference on the cloud.

<a id="the-computational-cost-of-building-ml-products"></a>

## The Computational Cost of Building ML Products
- <span style="color:blue; font-weight:bold;">Long training times:</span> Running a suite of an experiment on a decent CPU (e.g., a QuadCore i7, with 8GB RAM) can take upwards of 3 hours to days and even weeks for the algorithms to converge and produce a result set.
- <span style="color:blue; font-weight:bold;">ML modeling is iterative:</span> This computational lag is especially dire because getting a decent result requires several iterations of experiments either to tune the different parameters of the algorithm or to carry out various forms of feature engineering to achieve the desired classifier/ model that generalizes "optimally" to new examples.
- <span style="color:blue; font-weight:bold;">High-performant computer hardware is expensive:</span> On-premise high-end machines are expensive. Moreover, the technical skills required to build a cluster of commodity machines running a Spark/Hadoop cluster might be overwhelming and even sometimes a distraction from the ML task.

<div class="fig figcenter">
    <img src="/assets/ieee_ompi/long_processing_times.png" width="55%" height="55%">
    <div class="figcaption" style="text-align: center;">
        Long processing times
    </div>
</div>

<a id="why-gcp"></a>

## Why GCP?
- <span style="color:blue; font-weight:bold;">Technology leadership:</span> Google is a top technology leader in the internet space with a range of top web products such as Gmail, Youtube, and Google Maps to mention just a few. The aforementioned products generate, store and process tons of Terabytes of data each day from internet users around the world.
- <span style="color:blue; font-weight:bold;">Taming big data:</span> To deal with this significant data, Google have massive investments in processing and storage research and infrastructure. Google as of today boasts some of the most impressive data center designs in the world to support their computational demands and computing services.
- <span style="color:blue; font-weight:bold;">High-speed computation:</span> Google Cloud Platform makes available to the public lighting fast computational speed (it is getting faster!) and high-tech storage capabilities with extremely low latency (meaning minimal delays in data transfer) and high throughput (can be naively described as the time taken to complete a job). This is all glued together by state of the art networking technology/ infrastructure.
- <span style="color:blue; font-weight:bold;">Ease of use:</span> The storage and processing platform on which are built products like Gmail, Google Docs and the like, are now accessible to the public and available for everyone to utilize.

<div class="fig figcenter">
    <img src="/assets/ieee_ompi/google-cloud-platform.png" width="80%" height="80%">
    <div class="figcaption" style="text-align: center;">
        <!-- Google Cloud Platform -->
    </div>
</div>

<a id="gcp-product-and-service-offerings"></a>

## GCP Product and Service Offerings

<a id="cloud-compute"></a>

### Cloud Compute
Virtual machines running on Google’s data centers around the world. They include:
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/compute/">Compute engine:</a></span> virtual computing instances for custom processing.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/appengine/">App engine:</a></span> a cloud managed platform for developing and deploying web, mobile, and IoT app.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/kubernetes-engine/">Kubernetes engine:</a></span> orchestration manager for custom docker containers based on Kubernetes.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/container-registry/">Container registry:</a></span> private container storage.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/functions/">Serverless cloud functions:</a></span> cloud-based functions to connect or extend cloud services.

<div class="fig figcenter">
    <img src="/assets/ieee_ompi/cloud-compute.png" width="80%" height="80%">
    <div class="figcaption" style="text-align: center;">
        Cloud Compute
    </div>
</div>

<a id="cloud-storage"></a>

### Cloud Storage
Provide scalable and high-availability storage options for live and archival data within the cloud perimeter. Cloud storage is set-up to cater for elastic storage demands. The cloud storage products include:
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/storage/">Cloud storage:</a></span> general purpose storage platform).
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/sql/">Cloud SQL:</a></span> cloud-managed MySQL and Postgre SQL.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/bigtable/">Cloud BigTable:</a></span> NoSQL petabyte-sized storage.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/spanner/">Cloud Spanner:</a></span> scalable/ high availability transactional storage.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/datastore/">Cloud Datastore:</a></span> transactional NoSQL database.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/persistent-disk/">Persistent disks:</a></span> block-storage for Virtual Machines.

<div class="fig figcenter">
    <img src="/assets/ieee_ompi/cloud-storage.png" width="80%" height="80%">
    <div class="figcaption" style="text-align: center;">
        Cloud Storage
    </div>
</div>

<a id="big-data-analytics"></a>

### Big Data/ Analytics
Offers a range of serverless big data and analytics solutions for data warehousing, stream, and batch analytics, cloud-managed Hadoop ecosystems, cloud-based messaging systems and data exploration. The big-data services include:
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/bigquery/">Cloud BigQuery:</a></span> serverless analytics/ data warehousing platform.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/dataproc/">Cloud Dataproc:</a></span> fully-managed Hadoop/ Apache Spark infrastructure.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/dataflow/">Cloud Dataflow:</a></span> Batch/ Stream data transformation/ processing.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/dataprep/">Cloud Dataprep:</a></span> serverless infrastructure for cleaning unstructured/ structured data for analytics.
- <span style="color:blue; font-weight:bold;"><a href="https://datastudio.google.com/">Cloud Datastudio:</a></span> data visualization/ report dashboards.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/datalab/">Cloud Datalab:</a></span> managed Jupyter notebook for machine learning/ data analytics.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/pubsub/">Cloud Pub/Sub:</a></span> serverless messaging infrastructure.

<div class="fig figcenter">
    <img src="/assets/ieee_ompi/big-data-analytics.png" width="80%" height="80%">
    <div class="figcaption" style="text-align: center;">
        Big Data/ Analytics
    </div>
</div>

<a id="cloud-ai"></a>

### Cloud AI
Leverage pre-trained models for custom artificial intelligence tasks through the use of REST APIs. This is the same technoogy stack used by Google applications such as Google Translate, and Photos. Google Cloud AI services include:
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/automl/">Cloud AutoML:</a></span> train custom machine learning models leveraging transfer learning.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/ml-engine/">Cloud Machine Learning Engine:</a></span> for large-scale distributed training and deployment of machine learning models.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/natural-language/">Cloud Natural Language:</a></span> extract/ analyze text from documents.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/speech-to-text/">Cloud Speech API:</a></span> transcribe audio to text.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/vision/">Cloud Vision API:</a></span> classification/ segmentation of images.
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/translate/">Cloud Translate API:</a></span> translate from one language to another. 
- <span style="color:blue; font-weight:bold;"><a href="https://cloud.google.com/video-intelligence/">Cloud Video Intelligence API:</a></span> extract metadata from video files.

<div class="fig figcenter">
    <img src="/assets/ieee_ompi/cloud-ai.png" width="80%" height="80%">
    <div class="figcaption" style="text-align: center;">
        Cloud AI
    </div>
</div>