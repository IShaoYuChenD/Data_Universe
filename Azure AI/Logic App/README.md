Key Phrase Analysis Automation



Overview

This project automates the extraction and analysis of key phrases from newly uploaded claim documents using Azure Logic Apps and Text Analytics.



Workflow Summary

Trigger – The Logic App is activated when a new claim.txt file is uploaded to a designated Azure Storage container.



Extraction – The app retrieves the content of the new document.



Analysis – The text is sent to Azure Cognitive Services – Text Analytics to extract key phrases.



Storage – The analysis results are automatically saved in a separate output folder within Azure Storage.



Purpose

The automation streamlines claim processing by:

Eliminating manual text extraction

Providing quick insights from unstructured data

Enhancing efficiency and traceability within the claims workflow



Technologies Used

Azure Logic Apps

Azure Cognitive Services (Text Analytics API)

Azure Blob Storage

