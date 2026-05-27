---
license: apache-2.0
task_categories:
- question-answering
language:
- en
tags:
- multi-document reasoning
- entity disambiguation
- ambiguous QA
task_ids:
- open-domain-qa
size_categories:
- 10K<n<100K
pretty_name: AmbigDocs
source_datasets:
  - original
annotations_creators:
  - no-annotation
dataset_info:
- config_name: default
  features:
  - name: qid
    dtype: string
  - name: ambiguous_entity
    dtype: string
  - name: question
    dtype: string
  - name: documents
    sequence:
        - name: title
          dtype: string
        - name: text
          dtype: string
        - name: pid
          dtype: string
        - name: answer
          dtype: string
---

# Dataset Card for AmbigDocs

### Dataset Summary
AmbigDocs is a benchmark for testing the abilities of current LMs to distinguish confusing entity mentions and generate a cohesive answer.

### Language
English

### Data Fields

Each instance contains the following fields:
* `qid`: id of the data instance.
* `ambiguous_entity`: an entity that can be interpreted as any of disambiguated entities, depending on the context.
* `question`: question that contains the ambiguous entity.
* `documents`: list of documents, where each document contains the following fields.
  * `title`: title of the document, which is also the distinct entity that share the same ambiguous name.
  * `text`: text of the document, each containing an answer to the question.
  * `pid`: id of the document, from 2018-12-20 Wikipedia corpus in [DPR](https://github.com/facebookresearch/DPR).
  * `answer`: answer to the question, which can be inferred from the document.

### Data Splits
* `Train`: 25268
* `Validation`: 3610
* `Test`: 7220

### Dataset Creation
Please refer to our [paper](https://arxiv.org/abs/2404.12447) (Section 3) for details on annotation process and discussion on limitations.