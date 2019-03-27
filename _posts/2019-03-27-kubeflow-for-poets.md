---
layout: post
title: 'Kubeflow for Poets: A Guide to Containerization of the Machine Learning Production Pipeline'
date: 2019-03-27 10:56:00 +00:00
category: Writings
comments: false
excerpt: "This writing series provides a systematic approach to productionalizing machine learning pipelines with Kubeflow on Kubernetes. Building machine learning models is just one piece of a more extensive system of tasks and processes that come together to deliver a Machine Learning product. Kubeflow makes it possible to leverage the microservices paradigm of containerization to separate modular components of an application orchestrated on Kubernetes."
permalink: /kubeflow-for-poets/
---

<p align="left">
<img src="../assets/kubeflow_poets/docker.png" align="middle" alt="Docker." height="20%" width="20%"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="../assets/kubeflow_poets/kubernetes.jpg" align="middle" alt="Kubernetes." height="20%" width="20%"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="../assets/kubeflow_poets/kubeflow.jpg" align="middle" alt="Kubeflow." height="20%" width="20%"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="../assets/kubeflow_poets/gcp.png" align="middle" alt="Google Cloud Platform." height="20%" width="20%"/>
</p>

<br>
This writing series provides a systematic approach to productionalizing machine learning pipelines with Kubeflow on Kubernetes. Building machine learning models is just one piece of a more extensive system of tasks and processes that come together to deliver a Machine Learning product. Kubeflow makes it possible to leverage the microservices paradigm of containerization to separate modular components of an application orchestrated on Kubernetes. While Kubernetes is platform agnostic, this series will focus on deploying a Machine Learning product on Google Cloud Platform leveraging Google Cloud BigQuery, Google Cloud Dataflow and Google Cloud Machine Learning Engine orchestrated on Google Kubernetes Engine.

## Contents:
The content is arranged as follows:
- <a href="/introduction-kubeflow-for-poets">Introduction</a>
- <a href="/microservices-kubeflow-for-poets">Microservices Architecture</a>
- <a href="/docker-kubeflow-for-poets">Docker</a>
- <a href="/kubernetes-kubeflow-for-poets">Kubernetes</a>
- <a href="/kubeflow-kubeflow-for-poets">Kubeflow</a>
- <a href="/kubeflow-pipelines-kubeflow-for-poets">Kubeflow Pipelines</a>
- <a href="/end-to-end-kubeflow-pipelines-kubeflow-for-poets">Deploying an End-to-End Machine Learning Solution on Kubeflow Pipelines</a>

## Links:
 - <a href="https://www.docker.com/">Docker</a>
 - <a href="https://kubernetes.io/">Kubernetes</a>
 - <a href="https://github.com/kubeflow/kubeflow">Kubeflow</a>
 - <a href="https://github.com/kubeflow/pipelines">Kubeflow Pipelines</a>
 - <a href="https://cloud.google.com/bigquery/">Google Cloud BigQuery</a>
 - <a href="https://cloud.google.com/dataflow/">Google Cloud Dataflow</a>
 - <a href="https://cloud.google.com/ml-engine/">Google Cloud Machine Learning Engine</a>
 - <a href="https://cloud.google.com/kubernetes-engine/">Google Kubernetes Engine</a>

## Source Code and Contribution:
The entire source code is available on <a href="https://github.com/dvdbisong/kubeflow-for-poets">Github</a>. Contributions and corrections are welcomed as pull requests.