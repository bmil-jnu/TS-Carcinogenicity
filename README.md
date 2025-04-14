# TS-Carcinogenicity

Title: **Tissue-Specific Carcinogenicity Prediction Using Multi-Task Learning on Attention-based Graph Neural Networks**

Authors: Yunju Song, Sunyong Yoo

## Description

We present a model that can provide interpretable predictions of tissue-specitif carcinogenicity of compounds using various multi-task learning approaches.
You can find the data and source code used in the paper.
In addition, we provide a Python file that can be used to generate predictions from the model we trained.
If you want, you can train and predict new datasets from the structure of the model we proposed.

- **[Dataset](https://github.com/bmil-jnu/TS-Carcinogenicity/tree/main/data)** : The dataset used in the paper.
- **[Multi task model source codes](https://github.com/bmil-jnu/TS-Carcinogenicity/tree/main/model/multi_task)** : Multi-task model training and gradient similarity calculation 
- **[Prediction analysis codes](https://github.com/bmil-jnu/TS-Carcinogenicity/tree/main/model/multi_task)** : Analytics for performance evaluation and attention highlighting
- **[Comparison model(single task) codes](https://github.com/bmil-jnu/TS-Carcinogenicity/tree/main/model/single_task)** : Performance evaluation comparison model (single task)
- **[Comparison model(CarcGC) codes](https://github.com/bmil-jnu/TS-Carcinogenicity/tree/main/model/CarcGC)** : Performance evaluation comparison model (CarcGC)
- **[Comparison model(DCAMCP) codes](https://github.com/bmil-jnu/TS-Carcinogenicity/tree/main/model/DCAMCP)** : Performance evaluation comparison model (DCAMCP)

## Dependency

`Pillow==9.5.0`
`iterative-stratification==0.1.7`
`pandas==1.1.5`
`rdkit==2023.3.1`
`scikit-learn==1.0.2`
`torch==1.12.1+cu113`
`torch-geometric==2.3.1`


# Contacts

If you have any questions or comments, please feel free to create an issue on github here, or email us:

- syunju0814@gmail.com
- syyoo@jnu.ac.kr
