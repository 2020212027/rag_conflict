---
dataset_info:
  features:
  - name: qid
    dtype: string
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: subset
    dtype: string
  - name: justification
    dtype: string
  splits:
  - name: test
    num_bytes: 6874752
    num_examples: 1500
  download_size: 2650076
  dataset_size: 6874752
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---

# FaithEval

FaithEval is a new and comprehensive benchmark dedicated to evaluating contextual faithfulness in LLMs across three diverse tasks: unanswerable, inconsistent, and counterfactual contexts.

[Paper] FaithEval: Can Your Language Model Stay Faithful to Context, Even If "The Moon is Made of Marshmallows", ICLR 2025, https://arxiv.org/abs/2410.03727

[Code and Detailed Instructions] https://github.com/SalesforceAIResearch/FaithEval


## Disclaimer and Ethical Considerations
This release is for research purposes only in support of an academic paper. Our datasets and code are not specifically designed or evaluated for all downstream purposes. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people’s lives, rights, or safety. For further guidance on use cases, refer to our AUP and AI AUP.
