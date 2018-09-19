---
layout: page-seminar
title: 'Setting Up an account on Google Cloud Platform'
permalink: gcp-ml-seminar/setting-up-gcp-account/
---

Table of contents:

- [Creating an Account](#creating-an-account)
- [GCP Resources: Projects](#gcp-resources-projects)
- [Accessing Cloud Platform Services](#accessing-cloud-platform-services)
- [Account Users and Permissions](#account-users-and-permissions)
- [The Cloud Shell](#the-cloud-shell)

This post walks through setting up a Google Cloud Platform account. A GCP account gives access to all of the platform's infrastructure and services. For a new account, a $300 credit is awarded, to be spent over a period of 12 months. This offer is great as it gives ample time to explore the different features and services of Google's cloud offering.

Note that to register an account requires a valid credit card to validate that it is an authentic user, as opposed to a robot. However, the credit card won't be charged after the trial ends, except Google is authorized to do so.

### Creating an Account
1. Go to https://cloud.google.com/ to open an account
   
   <div class="fig figcenter fighighlight">
     <img src="/assets/seminar_IEEE/GCP-login-page.png">
     <div class="figcaption" style="text-align: center;">
        Figure 1: Google Cloud Platform Login Page.
     </div>
    </div>

2. Fill in the necessary identity, address and credit card details.
3. Wait a moment, while an account is created on the platform.

    <div class="fig figcenter fighighlight">
     <img src="/assets/seminar_IEEE/creating-account.png">
     <div class="figcaption" style="text-align: center;">
        Figure 2: Creating Account.
     </div>
    </div>

4. After account creation, we're presented with the Welcome to GCP page.

    <div class="fig figcenter fighighlight">
     <img src="/assets/seminar_IEEE/welcome-to-GCP.png">
     <div class="figcaption" style="text-align: center;">
        Figure 3: Welcome to GCP.
     </div>
    </div>

5. Click on the triple-dash in the top-right corner of the page <span style="color:green">(1)</span>, then click on <span style="color:green">Home (2)</span> to open the Google Cloud Platform Dashboard.

    <div class="fig figcenter fighighlight">
     <img src="/assets/seminar_IEEE/GCP-dashboard.png">
     <div class="figcaption" style="text-align: center;">
        Figure 4: GCP Dashboard.
     </div>
    </div>

The Cloud Dashboard provides a birds-eye summary of the project such as the current billing rate and other resource usage statistics. The activity tab to the right gives a breakdown of the resource actions performed on the account. This feature is useful when building an audit trail of events.

### GCP Resources: Projects
All the services and features of the Google cloud platform are called resources. These resources are arranged in a hierarchical order, with the top-level being the project. The project is like a container that houses all GCPs resources. Billing on an account is attached to a project. Multiple projects can be created for an account.

To view the projects in the account, click on the <span style="color:green">scope picker in the cloud console</span> (see red highlight in figure below)

<div class="fig figcenter fighighlight">
     <img src="/assets/seminar_IEEE/project-scope-picker.png">
     <div class="figcaption" style="text-align: center;">
        Figure 5: Scope Picker to select projects.
     </div>
     <img src="/assets/seminar_IEEE/select-projects.png">
     <div class="figcaption" style="text-align: center;">
        Figure 6: Select projects.
     </div>
</div>

### Accessing Cloud Platform Services
To access the resources on the cloud platform, click the triple-dash in the top-right corner of the window. Grouped service offerings are used to organize the resources. For example, in the image below, we can see the products under <span style="color:green">Storage:</span> Bigtable, Datastore, Storage, SQL, and Spanner.

<div class="fig figcenter fighighlight">
     <img src="/assets/seminar_IEEE/accessing-resources.png">
     <div class="figcaption" style="text-align: center;">
        Figure 7: Google Cloud Platform Services.
     </div>
</div>

### Account Users and Permissions
GCP allows you to define security roles and permission for every resource in a specific project. This feature is particularly useful when a project scales beyond one user. New roles and permissions are created for a user through the IAM & Admin tab.

<div class="fig figcenter fighighlight">
     <img src="/assets/seminar_IEEE/open-IAM.png">
     <div class="figcaption" style="text-align: center;">
        Figure 8: Open IAM & Admin.
     </div>
     <img src="/assets/seminar_IEEE/IAM-console.png">
     <div class="figcaption" style="text-align: center;">
        Figure 9: IAM & Admin Console.
     </div>
</div>

### The Cloud Shell
The Cloud shell is a vital component for working with GCP resources. It gives the user cloud-based command-line access to manipulate resources directly from the platform without installing the Google Cloud SDK on a local machine.

The cloud shell is accessed by clicking on the <span style="color:green">prompt icon</span> in the top-left corner of the window.

<div class="fig figcenter fighighlight">
     <img src="/assets/seminar_IEEE/activate-cloud-shell.png" width="60%" height="60%">
     <div class="figcaption" style="text-align: center;">
        Figure 10: Activate Cloud Shell.
     </div>
     <img src="/assets/seminar_IEEE/start-cloud-shell.png">
     <div class="figcaption" style="text-align: center;">
        Figure 11: Start Cloud Shell.
     </div>
     <img src="/assets/seminar_IEEE/cloud-shell-interface.png">
     <div class="figcaption" style="text-align: center;">
        Figure 12: Cloud Shell Interface.
     </div>
</div>