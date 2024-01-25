# Music-Genre-Classification-Using-Azure-Machine-Learning-Studio

 This project focuses on applying Convolutional Neural Networks (CNNs) to the FMA (Free Music Archive) Small dataset. The FMA dataset provides a diverse collection of audio tracks. The small dataset consists of 8000 tracks of the 30s, 8 balanced genres (https://os.unil.cloud.switch.ch/fma/fma_small.zip). The main purpose of this project is to provide A comprehensive guide on deploying and running the project on Azure Machine Learning Studio since this dataset needs at least 30 GB of RAM.

 **1 Setting Up Resources**

 The first step is creating an Azure account to go ahead. after signing into the portal, click on Create a Resource. From options simply click on the add button to create a new machine learning workspace. workspace name, subscription, resource group, and region should be filled. before moving to the studio, the dataset should be added to Azure because uploading the dataset directly to ML studio is time-consuming. Similarly, in the portal click on add to create Storage Account. inside the Storage directory navigate to the container section and create a new container like fmadataset to store your dataset. For uploading data Azure Storage Explorer could be used. you need to connect to your subscription from Storage Explorer. The string password can be found in the access key inside your storage account.

**2 Running Project on Azure Machine Learning Studio**

In the Azure ML directory go to compute at the bottom of the page and click on new. Here you can choose your desired RAM and storage. For this Project, 32 GB of RAM and 150 GB of disk space is enough. The most crucial consideration in this step is to set up a connection between the storage account and ML studio. In the assets panel, click on Data and then create a new datastore. datastore must be connected to your dataset in the container of the Storage Account. If you cannot connect, go back to your Storage Account and from access control check whether your instance has enough permission to transfer the data or not. after implementing all of the steps successfully, your portal in studio should look like the below photo:
![image](https://github.com/nourihouman/Music-Genre-Classification-Using-Azure-Machine-Learning-Studio/assets/150868916/e3a0e50e-923f-4daf-9e37-5cfd2822c62f)
