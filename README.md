# Aerial Image Classification

> **NOTE** This content is no longer maintained. Visit the [Azure Machine Learning Notebook](https://github.com/Azure/MachineLearningNotebooks) project for sample Jupyter notebooks for ML and deep learning with Azure Machine Learning.

## Link to the Microsoft DOCS site

The detailed documentation for this real world scenario includes the step-by-step walkthrough:

[https://docs.microsoft.com/azure/machine-learning/preview/scenario-aerial-image-classification](https://docs.microsoft.com/azure/machine-learning/preview/scenario-aerial-image-classification)

## Link to the Gallery GitHub repository

The public GitHub repository for this real world scenario contains all the code samples:
[https://github.com/Azure/MachineLearningSamples-AerialImageClassification](https://github.com/Azure/MachineLearningSamples-AerialImageClassification)

## Overview

In this scenario, we train machine learning models to classify the type of land shown in aerial images of 224-meter x 224-meter plots. Land use classification models can be used to track urbanization, deforestation, loss of wetlands, and other major environmental trends using periodically collected aerial imagery. After training and validating the classification model, we will apply it to aerial images spanning Middlesex County, MA -- home of Microsoft's New England Research & Development (NERD) Center -- to demonstrate how these models can be used to study trends in urban development. This example includes two approaches for distributed model training with Azure Machine Learning (AML) Workbench: deep neural network training on [Azure Batch AI](https://batchaitraining.azure.com/) GPU clusters, and transfer learning using the [Microsoft Machine Learning for Apache Spark (MMLSpark)](https://github.com/Azure/mmlspark) package. The example concludes with an illustration of model operationalization for scoring large static image sets on an [Azure HDInsight Spark](https://azure.microsoft.com/en-us/services/hdinsight/apache-spark/) cluster.

## Key components needed to run this scenario
- An [Azure account](https://azure.microsoft.com/en-us/free/) (free trials are available), which will be used to create an HDInsight Spark cluster with 40 worker nodes and an Azure Batch AI GPU cluster with two VMs/two GPUs.
- [Azure Machine Learning Workbench](https://review.docs.microsoft.com/en-us/azure/machine-learning/preview/overview-what-is-azure-ml).
- [AzCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy), a free utility for coordinating file transfer between Azure storage accounts.
- An SSH client; we recommend [PuTTy](http://www.putty.org/).

## Data/Telemetry
Aerial Image Classification collects usage data and sends it to Microsoft to help improve our products and services. Read our [privacy statement](http://go.microsoft.com/fwlink/?LinkId=521839) to learn more. 

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (for example, label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
