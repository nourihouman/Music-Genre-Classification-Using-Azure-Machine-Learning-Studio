# Music-Genre-Classification-Using-Azure-Machine-Learning-Studio

 This project focuses on applying Convolutional Neural Networks (CNNs) to the FMA (Free Music Archive) Small dataset. The FMA dataset provides a diverse collection of audio tracks. Small dataset is consist of 8000 tracks of 30s, 8 balanced genres (https://os.unil.cloud.switch.ch/fma/fma_small.zip). The main purpose of this project is to provide A comprehensive guide on deploying and running the project on Azure Machine Learning Studio since this dataset needs at least 30 Gb of RAM.

 **1.1 Setting Up Resources**

 The first step is creating an Azure account to go ahead. after signing into the portal, click on Create a Resource. From options simply click on the add button to create a new machine learning workspace. workspace name, subscription, resource group, and region should be filled. before moving to the studio, the dataset should be added to Azure because uploading dataset directly to ML studio is time-consuming. Similarly, in the portal click on add to create Storage Account. inside the Storage directory navigate to the container section and create a new container like fmadataset to store your dataset. For uploading data Azure Storage Explorer could be used. you need to connect to your subscription from Storage Explorer. The string password can be found in the access key inside your storage account.

 Set Up Azure Storage Account
In the Azure Portal, navigate to the Storage Accounts section.
Click on "+ Add" to create a new storage account.
Provide a unique name, choose the desired configuration, and complete the creation process.
Step 4: Upload Dataset to Azure Blob Storage
In the Azure Portal, go to the Storage Account you just created.
Navigate to the "Containers" section and create a new container to store your dataset.
Upload your dataset to the created container. You can use the Azure Portal, Azure Storage Explorer, or Azure Storage Explorer in VS Code for this![image](https://github.com/nourihouman/Music-Genre-Classification-Using-Azure-Machine-Learning-Studio/assets/150868916/558fb8d7-8318-4826-b359-3c744384ae48)
