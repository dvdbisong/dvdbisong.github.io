---
layout: post
title: 'Exploring Amazon SageMaker'
date: 2018-01-14 04:15:19 +00:00
category: Technical Writings
comments: true
excerpt: "Amazon SageMaker is another cloud-based fully managed data analytics/ machine learning modeling platform for designing, building and deploying data models. The key selling point of Amazon SageMaker is \"zero-setup\". This post takes a tour through spinning up a SageMaker notebook instance for data analytics/ modeling learning models."
permalink: /exploring-amazon-sagemaker/
image: ../assets/exploring_sagemaker/1.SM_home.png
---

Amazon SageMaker is another cloud-based fully managed data analytics/ machine learning modeling platform for designing, building and deploying data models. The key selling point of Amazon SageMaker is "zero-setup". The concept of "zero-setup" means data science teams can entirely focus on building the model without concern for infrastructure configuration. Also, SageMaker makes available some pre-configured black-boxed modules to streamline the building, training, and deployment of machine learning models. The goal is to further democratize machine learning for software developers without cutting-edge ML expertise.

Amazon SageMaker deploys cloud-based Jupyter notebooks for rapid machine learning design and development. These notebooks are pre-configured to run machine learning libraries like <span style="font-style: italic; color:green">TensorFlow</span> and <span style="font-style: italic; color:green">MXNet</span>. You can link to your data stored on Amazon S3, an Amazon "general purpose" cloud data storage system. As mentioned earlier, SageMaker offers a pre-configured and easy one-click system to build Machine Learning models from your dataset.

In this tutorial, we'll walk you through getting started working with Amazon SageMaker. To begin, please [click here to sign-in to your AWS account](https://console.aws.amazon.com/sagemaker). If you do not have one, you can create a new account before proceeding. The SageMaker dashboard is shown below.

<div class="imgcap">
<img src="../assets/exploring_sagemaker/1.SM_home.png">
<div class="thecap"><span style="font-style: italic">Figure 1: </span>Amazon SageMaker Dashboard</div>
</div>

### Create an S3 Bucket
We'll first need a data storage location to store the data for training and evaluating the model. Also, SagerMaker will require a specified bucket to save the contents of the model during the training process. To later give the SageMaker notebook instance access to the S3 bucket, append the word <span style="font-style: italic; color:red">'sagemaker'</span> to the designated bucket name.

The notebook instance is assigned a default <span style="font-family: Courier New,Courier,Lucida Sans Typewriter,Lucida Typewriter,monospace">AmazonSageMakerFullAccess</span> IAM role (unless changed in the configuration). [To create an S3 bucket, click here.](https://console.aws.amazon.com/s3/)

<div class="imgcap">
<img src="../assets/exploring_sagemaker/3.create_S3_bucket.png">
<div class="thecap"><span style="font-style: italic">Figure 2: </span>Create S3 Bucket</div>
</div>

### Create a NoteBook Instance
Amazon SageMaker spins up a cloud-managed Jupyter notebook instance for prototyping, exploring, pre-processing, testing and evaluating your models. To create a notebook instance, click <span style="font-style: italic; color:red">'create notebook instance'</span> within the red box highlight in *Figure 1*. The notebook instance configuration page is shown in *Figure 3*.

<div class="imgcap">
<img src="../assets/exploring_sagemaker/2.instance_configuration.png">
<div class="thecap"><span style="font-style: italic">Figure 3: </span>Notebook Instance Configuration</div>
</div>

#### Create an <span style="font-family: Courier New,Courier,Lucida Sans Typewriter,Lucida Typewriter,monospace">IAM</span> Role
Give the notebook instance a name and select an instance type (the default setting is sufficient). To complete the notebook instance configuration, click <span style="font-style: italic; color:red">'create a new role'</span> to create an <span style="font-family: Courier New,Courier,Lucida Sans Typewriter,Lucida Typewriter,monospace">IAM</span> role as shown in *Figure 4* below.

<div class="imgcap">
<img src="../assets/exploring_sagemaker/4.create_IAM.png">
<div class="thecap"><span style="font-style: italic">Figure 4: </span>Create <span style="font-family: Courier New,Courier,Lucida Sans Typewriter,Lucida Typewriter,monospace">IAM</span> Role</div>
</div>

Click <span style="font-style: italic; color:red">'create role'</span> to return to the instance configuration page (*Figure 5*).

<div class="imgcap">
<img src="../assets/exploring_sagemaker/5.create_notebook.png">
<div class="thecap"><span style="font-style: italic">Figure 5: </span>Create Notebook</div>
</div>

Click <span style="font-style: italic; color:red">'create notebook instance'</span> to spin up the compute engine. The new instance is spurned and shown on the instance page (*Figure 6*). Wait for the status to go from <span style="font-style: italic; color:green">Pending</span> to <span style="font-style: italic; color:green">InService</span>. Then the instance is ready to run a Jupyter notebook.

<div class="imgcap">
<img src="../assets/exploring_sagemaker/6.notebook_instance.png">
<div class="thecap"><span style="font-style: italic">Figure 6: </span>Notebook Instance Page</div>
</div>

#### Open the (Jupyter) Notebook Instance
Select the created instance and click <span style="font-style: italic; color:red">'Open'</span> to open the instance running Jupyter notebook (*Figure 7*).

<div class="imgcap">
<img src="../assets/exploring_sagemaker/7.open_the_notebook.png">
<div class="thecap"><span style="font-style: italic">Figure 7: </span>Open the Notebook</div>
</div>

The Jupyter notebook console is shown in *Figure 8*. From the image below SageMaker comes with sample notebooks to get started with using the out-of-the-box Amazon optimized machine learning models, as well as samples for TensorFlow and MxNet models, and getting started with advanced functionalities like loading an already developed model to SageMaker and setting up an application endpoint.

<div class="imgcap">
<img src="../assets/exploring_sagemaker/8.notebook_console.png">
<div class="thecap"><span style="font-style: italic">Figure 8: </span>Notebook Console</div>
</div>

Let's create a new notebook and run some TensorFlow code to ensure that all is working as anticipated. Observe that we can open notebooks pre-configured with PySpark, Apache Spark, MxNet, and TensorFlow. We'll open a TensorFlow python3 notebook.

<div class="imgcap">
<img src="../assets/exploring_sagemaker/9.running_tensorflow.png">
<div class="thecap"><span style="font-style: italic">Figure 8: </span>Running TensorFlow</div>
</div>

So there you have it, a fully managed, distributed and serverless environment for machine learning modeling and deployment.