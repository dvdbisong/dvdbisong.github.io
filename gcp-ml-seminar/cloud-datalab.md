---
layout: page-seminar
title: 'Google Datalab'
permalink: gcp-ml-seminar/cloud-datalab/
---

Google Cloud datalab provides an interactive environment (via Jupyter notebooks) for analyzing data and designing machine learning models. Datalab is actually hosted on a Google Compute Engine with an attached disk capacity for storing the Jupyter notebooks.

In this piece, we'll be launching Cloud Datalab for machine learning design using the gcloud command-line console. 

### Enable Google Compute Engine and Cloud Source Repositories APIs
On a new account, first, enable Google Compute API and then the Cloud Source Repositories API before launching datalab.

<div class="fig figcenter fighighlight">
     <img src="/assets/seminar_IEEE/api-dashboard.png">
     <div class="figcaption" style="text-align: center;">
        Figure 1: APIs & Services Dashboard
     </div><br>
     <span style="color:green; font-weight: bold">Enable Cloud Compute Engine API</span>
     <img src="/assets/seminar_IEEE/enable-api.png">
     <div class="figcaption" style="text-align: center;">
        Figure 2: Search for Compute Engine APIs 
     </div>     
     <img src="/assets/seminar_IEEE/enable-compute-api.png">
     <div class="figcaption" style="text-align: center;">
        Figure 3: Enable Compute API
     </div>
     <img src="/assets/seminar_IEEE/compute-engine-api.png">
     <div class="figcaption" style="text-align: center;">
        Figure 4: Google Compute Engine API Dashboard
     </div><br>
     <span style="color:green; font-weight: bold">Enable Cloud Source Repositories API</span>
     <img src="/assets/seminar_IEEE/find-cloud-source-repo.png">
     <div class="figcaption" style="text-align: center;">
        Figure 5: Search for Cloud Source Repositories APIs 
     </div>     
     <img src="/assets/seminar_IEEE/enable-cloud-source-repo.png">
     <div class="figcaption" style="text-align: center;">
        Figure 6: Enable Cloud Source Repositories API
     </div>
     <img src="/assets/seminar_IEEE/cloud-source-repo-dashboard.png">
     <div class="figcaption" style="text-align: center;">
        Figure 7: Google Cloud Source Repositories API Dashboard
     </div>
</div>

### Launch a Datalab Instance from gcloud
Working with the gcloud console allows us to interact with Google Cloud resources using the command-line directly from the browser. This tool is very convenient because the user need not install the Google Cloud SDK on the local machine. The gcloud feature allows users of low-end computers to do a lot of resource interaction directly within the cloud perimeter.

To launch a datalab instance,
1. Open gcloud shell
2. Create a datalab instance by running the command <span style="color:red; font-family: Courier New,Courier,Lucida Sans Typewriter,Lucida Typewriter,monospace">datalab create instance-name</span>
3. Accept all the defaults as the instance is provisioned
4. Press <span style="color:green; font-family: Courier New,Courier,Lucida Sans Typewriter,Lucida Typewriter,monospace">Enter</span> twice when asked to generate a passphrase.
5. Click on the <span style="color:green;">Web Preview</span> at top-right of the gcloud window, click <span style="color:green;">Change port</span> and selet <span style="color:red;">Port 8081</span>, to start using Datalab.

**Note:** The <span style="color:red;">instance-name</span> must begin with a lowercase letter, and can have up to 63 lowercase letters, mixed with numbers and hyphens. The instance-name cannot end with a hyphen.

<div class="fig figcenter fighighlight">
     <img src="/assets/seminar_IEEE/activate-cloud-shell.png" width="60%" height="60%">
     <div class="figcaption" style="text-align: center;">
        Figure 8: Open Cloud Shell
     </div><br>
     <img src="/assets/seminar_IEEE/create-datalab.png">
     <div class="figcaption" style="text-align: center;">
        Figure 9: Create Datalab Instance
     </div><br>
     <img src="/assets/seminar_IEEE/select-zone.png">
     <div class="figcaption" style="text-align: center;">
        Figure 10: Select Zone
     </div><br>
     <img src="/assets/gcp_ml/11.press-Y.png">
     <div class="figcaption" style="text-align: center;">
        Figure 11: Type <span style="color:red;">Y</span> to generate SSH keys
     </div><br>
     <img src="/assets/seminar_IEEE/select-port-8081.png">
     <div class="figcaption" style="text-align: center;">
        Figure 12: Select Port 8081
     </div><br>
     <img src="/assets/seminar_IEEE/datalab-interface.png">
     <div class="figcaption" style="text-align: center;">
        Figure 13: Datalab Interface
     </div>
</div>

### Open a notebook
To open a new notebook. Click on <span style="color:red;">New Notebook</span> at the top-left hand corner of the instance page.

<div class="fig figcenter fighighlight">
     <img src="/assets/seminar_IEEE/new-notebook.png">
     <div class="figcaption" style="text-align: center;">
        Figure 14: New Notebook
     </div>
</div>

### Shutting down/ Deleting the instance
To shut down the instance,
1. Open the Cloud Compute Dashboard
2. Click the check-box to select the instance.
3. Click <span style="color:red;">Stop</span> to shut down the instance. Shutting down the instance prevents the user from incurring unnecessary cost when the instance is not in use.
4. Click <span style="color:red;">Delete</span> to delete the instance and detach the disk allocated to the compute engine. Only use this action when completely done with all work on that instance. This action is irreversible.

<div class="fig figcenter fighighlight">
     <img src="/assets/seminar_IEEE/compute-IEEE-dashboard.png">
     <div class="figcaption" style="text-align: center;">
        Figure 15: Compute Dashboard
     </div><br>
     <img src="/assets/seminar_IEEE/stop-datalab.png">
     <div class="figcaption" style="text-align: center;">
        Figure 16: Shut-down Datalab Instance
     </div><br>
     <img src="/assets/seminar_IEEE/delete-datalab.png">
     <div class="figcaption" style="text-align: center;">
        Figure 17: Delete Datalab Instance
     </div>
</div>