# FetoML

Title: Tissue-Specific Carcinogenicity Prediction Using Multi-Task Learning on Attention-based Graph Neural Networks

Authors: Myeonghyeon Jeong, Sunyong Yoo

## Description

We present FetoML, a model that can provide interpretable predictions of fetal toxicity of drugs using various machine learning approaches.
You can find the data and source code used in the paper.
In addition, we provide a Python file that can be used to generate predictions from the model we trained.
If you want, you can train and predict new datasets from the structure of the model we proposed.

- [Dataset](https://github.com/bmil-jnu/FetoML/tree/main/Data)
- [Model source codes](https://github.com/bmil-jnu/FetoML/tree/main/Model%20Code/Model)
- [Hyperparameter optimization](https://github.com/bmil-jnu/FetoML/tree/main/Model%20Code/Hyperparameter%20Optimization)
- [Prediction analysis codes](https://github.com/bmil-jnu/FetoML/tree/main/Model%20Code/Analysis)

## Dependency

`Python == 3.7`
`tensorflow == 2.7.0`
`keras == 2.7.0`
`scikit-learn == 0.24.2`
`RDKit == 2021.09.3`

## How to use

### Predictions with already trained models

How the prediction of fetotoxicity of drugs is done through the model we proposed.

**Command:**
```
python FetoML.py -input input.csv -model model_name -output output_name
```

- `input.csv`: Fill in the csv file you want to predict. The columns in the csv file must contain 'name' and 'smiles'. Also, the location of the csv file must exist in the **'Data'** folder.
  
- `model`: Enter the model you want to use for your prediction. The model input must be chosen from 'LR', 'SVM', 'RF', 'ET', 'GBM', 'XGB', or 'NN'. Alternatively, you can enter 'all' or 'recommend'.
  
    - `all`: You can have results from all available models.
    - `recommend`: You can have results from the models recommended in the paper (ET and NN models).
    - Selecting more than one model to get results: Enter two or more models separated by ',' but cannot include `all` or `recommend`. Example: `RF,ET,NN`
  
- `output_name`: Enter a name to distinguish the file you'll receive as a result. The csv file will be output in the form of `[{output_name}]_{model_name}_predict_result.csv` in the **'Results'** folder.
    
**Example:**

```
python FetoML.py -input fetal_toxicity_Test.csv -model recommend -output sample
```
    
### Using hyperparameters from our proposed model to make predictions after training on a new training set

You can use it to train on new training data, or when you want to perform a comparison with the model structures we proposed.

```
python Retraining.py -train train.csv -test test.csv -name project_name -output output_name
```

- `train.csv`: Enter the dataset you want to train models on.  The columns in the csv file must contain 'name', 'smiles', and 'category'. Also, the location of the csv file must exist in the **'Data'** folder.
  </br>( **!!** Reference the `fetal_toxicity_Train.csv` file in the [Data](https://github.com/bmil-jnu/FetoML/tree/main/Data) folder )
  
- `test.csv`: Enter the name of csv file to predict. The columns in the csv file must contain 'name' and 'smiles'. Also, the location of the csv file must exist in the **'Data'** folder.

- `project_name`: Enter a name for your project. A new folder will be created with your project name. Subfolders in the project folder will contain the parameters of the trained models and the results for the testset.

- `output_name`: Enter a name to distinguish the file you'll receive as a result. The csv file will be output in the form of `[{output_name}]_{model_name}_predict_result.csv` in the **'Results'** folder.

**Example:**

```
python Retraining.py -train fetal_toxicity_Train.csv -test fetal_toxicity_Test.csv -name temp_project -output sample
```

# Contacts

If you have any questions or comments, please feel free to create an issue on github here, or email us:

- dureelee01@gmail.com
- syyoo@jnu.ac.kr
