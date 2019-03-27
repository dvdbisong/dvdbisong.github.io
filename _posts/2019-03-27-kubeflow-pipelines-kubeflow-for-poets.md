---
layout: post
title: 'Kubeflow Pipelines - Kubeflow for Poets'
date: 2019-03-27 12:06:00 +00:00
category: Writings
comments: false
excerpt: "Kubeflow Pipelines is a simple platform for building and deploying containerized machine learning workflows on Kubernetes. Kubeflow pipelines make it easy to implement production grade machine learning pipelines without bothering on the low-level details of managing a Kubernetes cluster."
permalink: /kubeflow-pipelines-kubeflow-for-poets/
---

Kubeflow Pipelines is a simple platform for building and deploying containerized machine learning workflows on Kubernetes. Kubeflow pipelines make it easy to implement production grade machine learning pipelines without bothering on the low-level details of managing a Kubernetes cluster.

Kubeflow Pipelines is a core component of Kubeflow and is also deployed when Kubeflow is deployed.

<div class="imgcap">
<img src="../assets/kubeflow_poets/kubeflow-pipelines-dashboard.png" alt="OAuth consent screen." height="90%" width="90%" />
<div>.</div>
</div>

## Components of Kubeflow Pipelines
A Pipeline describes a Machine Learning workflow, where each component of the pipeline is a self-contained set of codes that are packaged as Docker images. Each pipeline can be uploaded individually and shared on the Kubeflow Pipelines User Interface (UI). A pipeline takes inputs (parameters) required to run the pipeline and the inputs and outputs of each component.

The Kubeflow Pipelines platform consists of:
- A user interface (UI) for managing and tracking experiments, jobs, and runs.
- An engine for scheduling multi-step ML workflows.
- An SDK for defining and manipulating pipelines and components.
- Notebooks for interacting with the system using the SDK.
(Taken from: <a href="https://www.kubeflow.org/docs/pipelines/pipelines-overview/">Overview of Kubeflow Pipelines</a>)

## Executing a Sample Pipeline

1. Click on the name **[Sample] Basic - Condition**.

<div class="imgcap">
<img src="../assets/kubeflow_poets/select-a-simple-pipeline.png" alt="Select a simple pipeline." height="90%" width="90%" />
<div>.</div>
</div>

2. Click **Start an experiment**.

<div class="imgcap">
<img src="../assets/kubeflow_poets/create-an-experiment.png" alt="Create an experiment." height="90%" width="90%" />
<div>.</div>
</div>

3. Give the Experiment a Name.

<div class="imgcap">
<img src="../assets/kubeflow_poets/name-experiment.png" alt="Name the experiment." height="90%" width="90%" />
<div>.</div>
</div>

4. Give the Run Name.

<div class="imgcap">
<img src="../assets/kubeflow_poets/run-experiment.png" alt="Name the run." height="90%" width="90%" />
<div>.</div>
</div>

5. Click on the **Run Name** to start the Run.

<div class="imgcap">
<img src="../assets/kubeflow_poets/running-pipeline.png" alt="Running pipeline." height="90%" width="90%" />
<div>.</div>
</div>

## Delete Resources
See the end of <a href="/end-to-end-kubeflow-pipelines-kubeflow-for-poets">Deploying an End-to-End Machine Learning Solution on Kubeflow Pipelines</a> to delete billable GCP resources.