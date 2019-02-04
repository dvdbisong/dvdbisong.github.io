---
layout: page-ieee-ompi-workshop
title: 'Prototyping with Google Cloud Datalab'
permalink: ieee-ompi-workshop/datalab/
---

Table of contents:

- [Enable Google Compute Engine and Cloud Source Repositories APIs](#enable-google-compute-engine-and-cloud-source-repositories-apis)
- [Launch a Datalab Instance from gcloud](#launch-a-datalab-instance-from-gcloud)
- [Retrieving Code from Github into Datalab VM](#retrieving-code-from-github-into-datalab-vm)
- [Shutting down/ Deleting the instance](#shutting-down-deleting-the-instance)

Google Cloud datalab provides an interactive environment for model prototyping.

<a id="enable-google-compute-engine-and-cloud-source-repositories-apis"></a>

## Enable Google Compute Engine and Cloud Source Repositories APIs
On a new account, enable Google Compute API and then the Cloud Source Repositories API before launching datalab.

### APIs & Services Dashboard
<div class="fig">
    <img src="/assets/ieee_ompi/api-dashboard.png" width="80%" height="80%">
</div>

### Search for Compute Engine APIs
<div class="fig">
    <img src="/assets/ieee_ompi/enable-api.png" height="80%" width="80%">
</div>

### Enable Compute API
<div class="fig">
    <img src="/assets/ieee_ompi/enable-compute-api.png" alt="Enable Compute API." height="80%" width="80%" />
</div>

### Search for Cloud Source Repositories APIs
<div class="fig">
    <img src="/assets/ieee_ompi/find-cloud-source-repo.png" alt="Search for Cloud Source Repositories APIs." height="80%" width="80%" />
</div>

### Enable Cloud Source Repositories API
<div class="fig">
    <img src="/assets/ieee_ompi/enable-cloud-source-repo.png" alt="Enable Cloud Source Repositories API." height="80%" width="80%" />
</div>

<a id="launch-a-datalab-instance-from-gcloud"></a>

## Launch a Datalab Instance from gcloud
To launch a datalab instance,
1. Open gcloud shell
2. Create a datalab instance by running the command <span style="color:red; font-family: Courier New,Courier,Lucida Sans Typewriter,Lucida Typewriter,monospace">datalab create instance-name</span>
3. Accept all the defaults as the instance is provisioned
4. Press <span style="color:green; font-family: Courier New,Courier,Lucida Sans Typewriter,Lucida Typewriter,monospace">Enter</span> twice when asked to generate a passphrase.
5. Click on the <span style="color:green;">Web Preview</span> at top-right of the gcloud window, click <span style="color:green;">Change port</span> and selet <span style="color:red;">Port 8081</span>, to start using Datalab.

**Note:** The <span style="color:red;">instance-name</span> must begin with a lowercase letter, and can have up to 63 lowercase letters, mixed with numbers and hyphens. The instance-name cannot end with a hyphen.

### Open Cloud Shell
<div class="fig">
    <img src="/assets/ieee_ompi/activate-cloud-shell.png" alt="Open Cloud Shell." height="40%" width="40%" />
</div>

### Create Datalab Instance
<div class="fig">
    <img src="/assets/ieee_ompi/create-datalab.png" alt="Create Datalab Instance." height="80%" width="80%" />
</div>

### Select Zone
<div class="fig">
    <img src="/assets/ieee_ompi/select-zone.png" alt="Select Zone." height="80%" width="80%" />
</div>

### Generate SSH keys
<div class="fig">
    <img src="/assets/ieee_ompi/11.press-Y.png" alt="Generate SSH keys." height="80%" width="80%" />
</div>

### Select Port 8081
<div class="fig">
    <img src="/assets/ieee_ompi/select-port-8081.png" alt="Select Port 8081." height="80%" width="80%" />
</div>

### Datalab Interface
<div class="fig">
    <img src="/assets/ieee_ompi/datalab-interface.png" alt="Datalab Interface." height="80%" width="80%" />
</div>

### Open a notebook
To open a new notebook. Click on <span style="color:red;">New Notebook</span> at the top-left hand corner of the instance page.

<div class="fig">
    <img src="/assets/ieee_ompi/new-notebook.png" alt="Open a notebook." height="80%" width="80%" />
</div>

<a id="retrieving-code-from-github"></a>

## Retrieving Code from Github into Datalab VM
(1). Log-in into the Datalab VM via `ssh`.

<div class="fig">
    <img src="/assets/ieee_ompi/compute-engine-dashboard.png" alt="Compute Engine Dashboard." height="80%" width="80%" />
</div>

<div class="fig">
    <img src="/assets/ieee_ompi/select-vm-instance.png" alt="Select VM Instance." height="80%" width="80%" />
</div>

(2). Notebooks are stored in the disk location `/mnt/disks/datalab-pd/content/datalab/notebooks`. Change to the directory using the `cd` command and clone the repository from Github

#### `cd /mnt/disks/datalab-pd/content/datalab/notebooks`
<div class="fig">
    <img src="/assets/ieee_ompi/move-notebook-dir.png" alt="Move notebook directory." height="70%" width="70%" />
</div>

#### `git clone https://github.com/dvdbisong/IEEE-Carleton-and-OMPI-Machine-Learning-Workshop.git`
<div class="fig">
    <img src="/assets/ieee_ompi/git-clone.png" alt="Git clone repository." height="70%" width="70%" />
</div>

(3). The directory `/mnt/disks/datalab-pd/content` is mapped to the home directory in Datalab.

<div class="fig">
    <img src="/assets/ieee_ompi/notebook-with-repo.png" alt="Notebook with repository." height="70%" width="70%" />
</div>

<a id="shutting-down--deleting-the-instance"></a>

## Shutting down/ Deleting the instance
To shut down the instance,
1. Open the Cloud Compute Dashboard
2. Click the check-box to select the instance.
3. Click <span style="color:red;">Stop</span> to shut down the instance. Shutting down the instance prevents the user from incurring unnecessary cost when the instance is not in use.
4. Click <span style="color:red;">Delete</span> to delete the instance and detach the disk allocated to the compute engine. Only use this action when completely done with all work on that instance. This action is irreversible.

### Open Cloud Compute Dashboard
<div class="fig">
    <img src="/assets/ieee_ompi/compute-IEEE-dashboard.png" alt="Open Cloud Compute Dashboard." height="80%" width="80%" />
</div>

### Shut-down Datalab Instance
<div class="fig">
    <img src="/assets/ieee_ompi/stop-datalab.png" alt="Shut-down Datalab Instance." height="80%" width="80%" />
</div>

### Delete Datalab Instance
<div class="fig">
    <img src="/assets/ieee_ompi/delete-datalab.png" alt="Delete Datalab Instance." height="80%" width="80%" />
</div>
