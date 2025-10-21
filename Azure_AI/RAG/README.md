RAG-Based Semantic Search Setup



Overview

This project sets up a Retrieval-Augmented Generation (RAG) workflow for performing semantic search across multiple documents using Azure AI Foundry, Azure AI Search, and Chat Playground.



Workflow Summary

Deploy Embedding Model – Deploy text-embedding-3-small in Azure AI Foundry to generate vector representations of document content.



Set Up AI Search Index – Configure a RAG pipeline in Azure AI Search, create and populate an index with the embedded document data.



Connect LLM – In Chat Playground, deploy gpt-35-turbo as the language model and link it to the AI Search index for retrieval.



Test Semantic Search – Run test prompts using keywords or questions. The system retrieves and summarizes relevant files based on semantic similarity.



Purpose

This setup enables:

Intelligent document retrieval using semantic similarity instead of keyword matching

Context-aware responses by combining AI Search with GPT-based reasoning

Scalable integration of search and generation in enterprise applications



Technologies Used

Azure AI Foundry

Azure AI Search

Chat Playground (gpt-35-turbo)

text-embedding-3-small

