---
layout: page-seminar
title: 'Google Cloud Storage'
permalink: gcp-ml-seminar/cloud-storage/
---

Google cloud storage is a storage solution for virtually any type of data. Cloud storage is used to store both live and archival data and has guarantees of scalability (can store data of any size), consistency (the most updated version is served on request), durability (data is redundantly placed in separate geographic locations to eliminate loss), and high-availability (data is always available and accessible).

Let's take a brief tour through spinning and cleaning-up a storage bucket, as well as uploading to and deleting files (also called objects) from a cloud storage bucket.

### Create a bucket
A bucket is as the name implies, simply a container for storing data. A bucket is first created, before storing data in Google cloud storage.

To create a bucket,
1. Head over to the cloud storage dashboard and click <span style="color:green">'create bucket'</span>.
2. Give the bucket a unique name.
3. Select a 'bucket' storage class. A multi-region storage class is for buckets frequently accessed all over the world, whereas, the cold line storage is more or less for storing backup files. For now, the default selection is okay.
4. Click <span style="color:green">create</span> to set-up a bucket on Google Cloud Storage.

<div class="fig figcenter fighighlight">
     <img src="/assets/seminar_IEEE/cloud-storage-console.png">
     <div class="figcaption" style="text-align: center;">
        <span style="font-style: italic">Figure 1: </span>Cloud Storage Console
     </div>
     <img src="/assets/seminar_IEEE/create-a-bucket.png">
     <div class="figcaption" style="text-align: center;">
        <span style="font-style: italic">Figure 2: </span>Create a bucket
     </div>
</div>

### Uploading data to a bucket
Individual files or a complete folder can be uploaded into the bucket. As an example, let's upload a file from the local machine.

To upload a file to a cloud storage bucket on GCP,
1. Click <span style="color:green">'upload file'</span>.
2. Select the file from the file upload window, and click <span style="color:green">open</span>. 

<div class="fig figcenter fighighlight">
     <img src="/assets/seminar_IEEE/an-empty-bucket.png">
     <div class="figcaption" style="text-align: center;">
        <span style="font-style: italic">Figure 3: </span>An Empty Bucket
     </div>
     <img src="/assets/seminar_IEEE/uploading-files.png">
     <div class="figcaption" style="text-align: center;">
        <span style="font-style: italic">Figure 4: </span>Upload an object
     </div>
     <img src="/assets/seminar_IEEE/upload-successful.png">
     <div class="figcaption" style="text-align: center;">
        <span style="font-style: italic">Figure 5: </span>Upload successful
     </div>
</div>

### Delete objects from a bucket
Click the check-box beside the file and click <span style="color:green">delete</span>, to delete an object from a bucket.
<div class="fig figcenter fighighlight">
     <img src="/assets/seminar_IEEE/delete-file.png">
     <div class="figcaption" style="text-align: center;">
        <span style="font-style: italic">Figure 6: </span>Delete a file
     </div>
</div>

### Free up storage resource
To delete a bucket, or free-up a storage resource to prevent billing on a resource that is not used, click the check-box beside the bucket in question, and click <span style="color:green">delete</span> to remove the bucket and its contents. This action is not recoverable.
<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/select-bucket-to-delete.png">
     <div class="figcaption" style="text-align: center;">
        <span style="font-style: italic">Figure 7: </span>Select bucket to delete
     </div>
     <img src="/assets/seminar_IEEE/delete-bucket.png">
     <div class="figcaption" style="text-align: center;">
        <span style="font-style: italic">Figure 8: </span>Delete bucket
     </div>
</div>