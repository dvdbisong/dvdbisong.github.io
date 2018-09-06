---
layout: page-seminar
title: 'An Overview of Google Cloud Platform Services'
permalink: gcp-ml-seminar/gcp-services/
---

Table of contents:

- [Cloud Compute](#cloud-compute)
- [Cloud Storage](#cloud-storage)
- [Big Data/ Analytics](#big-data-analytics)
- [Cloud AI](#cloud-ai)
- [Skinny Data Migration Pipeline](#skinny-data-migration-pipeline)


Google Cloud Platform offers a wide range of services for securing, storing, serving and analyzing data. These cloud services form a secure cloud perimeter for data, where different operations and transformations can be carried out on the data without it ever leaving the cloud ecosystem. <!-- This data hub is sometimes referred to as a data lake. -->

The services offered by Google Cloud include Compute, Storage, Big Data/ Analytics, Artificial Intelligence (AI), and other networking, developer, and management services. Let's briefly review some of the features of the Google Cloud ecosystem.

<a name="cloud_compute"></a>

### Cloud Compute
Google Compute services are provisioned virtual machines that are physically running on Google's data centers around the world. The compute services consist of the compute engine (virtual computing instances for custom processing), app engine (a cloud managed platform for developing web, mobile, and IoT app), kubernetes engine (orchestration manager for custom docker containers based on Kubernetes), container registry (private container storage) and serverless cloud functions (cloud-based functions to connect or extend cloud services).

For our purposes of machine learning modeling, the cloud compute engine is most pertinent. As we will see later, a compute instance is provisioned that installs all the relevant tools/ packages and frameworks for training machine learning and deep learning models.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/cloud-compute.png"> <!--width="50%" height="50%"-->
    <div class="figcaption" style="text-align: center;">
        Figure 1: Cloud Compute Services.
    </div>
</div>

<a name="cloud_storage"></a>

### Cloud Storage
Google cloud storage options provide scalable and real-time storage access to live and archival data within the cloud perimeter. Cloud storage is set-up to cater for any conceivable storage demand. Data stored on Google cloud storage is available anytime, and from any location around the world. Whats more, this massive storage power comes at an almost negligible cost, taking into consideration the size and economic value of the stored data. Moreover, acknowledging the accessibility, security, and consistency provided by cloud storage, the cost is non-equivalent.

The cloud storage products include cloud storage (general purpose storage platform), cloud SQL (cloud-managed MySQL and Postgre SQL), cloud BigTable (NoSQL petabyte-sized storage), cloud spanner (scalable/ high availability transactional storage), cloud datastore (transactional NoSQL database), and persistent disks (block-storage for Virtual Machines).

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/cloud-storage.png">
    <div class="figcaption" style="text-align: center;">
        Figure 2: Cloud Storage Products
    </div>
</div>

<a name="big_data_analytics"></a>

### Big Data/ Analytics
Google Cloud Platform offers a range of serverless big data and analytics solutions for data warehousing, stream, and batch analytics, cloud-managed Hadoop ecosystems, cloud-based messaging systems and data exploration. These services provide multiple perspectives to mining/ generating real-time intelligence from big-data.

The big-data services include cloud BigQuery (serverless analytics/ data warehousing platform), cloud Dataproc (fully-managed Hadoop/ Apache Spark infrastructure), cloud Dataflow (Batch/ Stream data transformation/ processing), cloud Dataprep (serverless infrastructure for cleaning unstructured/ structured data for analytics), cloud Datastudio (data visualization/ report dashboards), cloud Datalab (managed Jupyter notebook for machine learning/ data analytics), and cloud Pub/Sub (serverless messaging infrastructure).

Cloud Datalab is our point of emphasis among the suite of services under the analytics category. Datalab is often used in conjunction with Cloud Storage for retrieving and storing data files and other programming assets, Cloud BigQuery for analytics of large-scale data from the interactive Jupyter notebook and Cloud ML for large-scale training and deployment of trained learning models.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/big-data-analytics.png">
    <div class="figcaption" style="text-align: center;">
        Figure 3: Big Data/ Analytics serverless platforms
    </div>
</div>

<a name="cloud_ai"></a>

### Cloud AI
Google Cloud AI offers a cloud service for businesses and individuals to leverage pre-trained models for custom artificial intelligence tasks through the use of REST APIs. This is the same service leveraged by notable Google applications such as Google Translate, Photos, and Inbox.

Google Cloud AI services include Cloud AutoML (train custom machine learning models leveraging transfer learning), Cloud Machine Learning Engine (for large-scale distributed training and deployment of machine learning models), Cloud Natural Language (extract/ analyze text from documents), Cloud Speech API (transcribe audio to text), Cloud Vision API (classification/ segmentation of images), Cloud Translate API (translate from one language to another), and Cloud Video Intelligence API (extract metadata from video files).

Cloud Machine Learning is often used in conjunction with Datalab to train TensorFlow models using distributed and highly optimized tensor processing units (TPUs). It is also used for deploying trained models for application consumption.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/cloud-ai.png">
    <div class="figcaption" style="text-align: center;">
        Figure 4: Cloud AI services
    </div>
</div>

<a name="data_migration"></a>

### Skinny Data Migration Pipeline
The critical pieces of the cloud platform under consideration with respect to designing and training machine learning models on data in the cloud is <span style="color:green">Google Cloud Datalab</span> and <span style="color:green">Google Cloud Storage</span>. The external data is transferred into the cloud perimeter and stored on cloud storage - from there it is loaded into cloud datalab for model design/ training using machine learning/ deep learning algorithms.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/ml-process.png" width="60%" height="60%">
    <div class="figcaption" style="text-align: center;">
        Figure 5: ML process
    </div>
</div>