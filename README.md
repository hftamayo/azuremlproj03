# Capstone Project: Homicides in El Salvador, comparing risk between men and women
### Machine Learning Engineer with Microsoft Azure Program
###### Scholarship recipient: Herbert Fernández Tamayo

#  Capstone Project: Homicides in El Salvador during 2018, comparing risk between men and women

### Table of Content
1. Project's overview
2. Project set up and installation
3. Dataset
- Overview
- Task
- Access
4. Automated ML
- Results
5. Hyperparameter tuning
- Results
6. Model deployment
7. Screen recording
8. Standout suggestions

## 1. Project's overview
As a part of the Machine Learning Engineer with Microsoft Azure Nanodegree, the third project is related to apply the knowledge acquired to solve or analyze a problem of the real life, it is important to choose a dataset related to a real scenario rather than a sample dataset, in this case I've chosen the Dataset of year 2018 related to Homicides in El Salvador, I work for the Statistical Department of the National Institute of Forensic Science of El Salvador, the institute is one of the most respected institution in the country related to the analysis about the behavior of crimes in El Salvador. The main goal of this experiment is to predict the incidence of homicides in males and females, with the results, it will be a point of start to plan policies of prevention in the cohort with a higher risk obtained from this experiment's result.

The project has two models: the first one is based on Automated Machine Learning method which consists in the evaluation of the dataset using multiples algorithms then we can choose the best one based on the level of accuracy obtained; the second one is based on HyperDrive method, the user sets hyperparameters to obtain results. Comparing the results of both models, it is encourage to deploy the best performing model through a web service and test it sending a data request.

The below diagram was provided by Capstone project's instructor with the idea to understand better how both models should be run:
![pdiagram.png](./img/pdiagram.png?raw=true "Project diagram")


## 2. Project Set Up and Installation
To run this project on your own Azure Machine Learning Studio environment you should follow the next steps:
1. Download from this repo the next files: automl.ipynb, hyperparameter_tuning.ipynb, train.py and cad2018.csv, you don't need to clone the entire repo.
2. Sign in your Azure ML Studio, upload the ipynb files and train.py
3. Create a type D2 Compute Instance, you may call it "notebooks"
4. From the new Compute Instance load Jupyter Notebook framework, from here, before open them load a terminal
5. By the time I run my experiments I need to update some dependencies, so from the terminal run the next command: pip install --upgrade azureml-sdk[notebooks,automl]
6. Next, open automl.ipynb and run each cell at a time, pay attention to the results
7. When step 6 has been finished you may open and run each cell on hyperparameter_tuning.ipynb file; again, pay attention to the results
8. If you want to try your own dataset, you may change the objects called: rawdata_homic2018 as well as the target column name in train.py and automl.ipynb

## 3. Dataset

### 3.1 Overview
The dataset, "cad2018.csv", contains the records of deaths by homicides in El Salvador during 2018, the data is gathered in 7 different offices around the country and monthly is validated by the Statistical Department, where I'm part of it,  of the National Institute of Forensic Science of El Salvador, since 2005 this country has been affected by the increase of deaths related to different way of violence, just in 2018 the official number of deaths related to homicides was 3,346.

The original dataset has more than 70 columns, for this experiment, I've chosen 7 columns because the other ones didn't have a close relation with the hypothesis I need to test; the information of each column ,and how each was recoded, is the next:

-id: internal ID field for each record, this is the primary key field. [no recoded]

-regfalle: national ID number assigned for each case of homicide. [no recoded]

-edad: age by the time the person has been murdered. [no recoded]

sexo: sex of the murdered person. Recoded values: [male -> 0, female -> 1]

-deptoocuhe: name of the state where the homicide was commited: Recoded values: [Ahuachapan->1, Santa Ana->2, Sonsonate->3, Chalatenango->4, La Libertad->5, San Salvador->6, Cuscatlan->7, La Paz->8, Cabanas->9, San Vicente->10, Usulutan->11, San Miguel->12, Morazan->13, La Union->14]

-tipoarma: name of the object used to commit the crime: Recoded values: [arma de fuego->1, asf x estrangulacion->2, asf x ahorcadura->3, asf x sofocacion->4, asf x sumersion->5, blanca sin espec->6, caida provocada->7, cortante->8, cortocontundente->9, cortopunzante->10, manos y pies->11, no datos->12, objeto contundente->13, lapidado->14, punzante->15, quemadura x fuego->16

pracaut: Authopsy practiced to the corpse: Recoded values: [si->1, no->0]

Any user may request a copy of the dataset, it is necessary to specify what variables are requested, also it is mandatory to fulfill the next form:
[Request information](https://transparencia.oj.gob.sv/es/solicitud-informacion)


### 3.2 Task
The main objective of this project is to run and find the best of two models, one using HyperDrive experiment and the other one using Automated Machine Learning experiment, that can help us to determine if the salvadoran male population will have a higher risk than salvadoran women of been murdered in homicides circumstances.

From 2010 in El Salvador there are different campaings to report and prevent murders just in women population -which is a great innitiative- but the purpose running this project is to put in the map that salvadoran male have almost thrice possilibites to died in homicides circumstances.

The column key -sexo- has this categories: 0->male, 1-> female, which resembles it is a classification problem. (ANCLA)


### 3.3 Access
For this experiment, the recoded data has been uploaded to this repository, from the jupyter notebook files the source code to access it is the next one:

from azureml.data.dataset_factory import TabularDatasetFactory

rawdata_homic2018 = "https://raw.githubusercontent.com/hftamayo/azuremlproj03/main/cad2018.csv"
dshomic2018 = TabularDatasetFactory.from_delimited_files(path=rawdata_homic2018, separator=',')


## 4. Automated ML
The configuration as well as other technical details related to the Automated ML experiment are the next ones:
-experiment_timeout_minutes : 20 (defines the exit criteria of each iteration)

-max_concurrent_iterations: 5 (number of thread -iteration- that can be executed simultaneously)

-primary_metric : 'accuracy' (this is the metric the experiment will try to optimize)

-task: 'classification' (key task that the experiment will focus to solve)

-label_column_name='sexo' --> key column the experiment will try to predict)


### 4.1 Results

The algorithm with the best performance during the tests was "VotingEnsemble" with a score of 0.8822, for a detailed list of the results you may check the file "automl.ipynb" section "Run Details"; in the next picture we can see the last algorithms executed and its results:

![automl_01.png](./img/automl_01.png?raw=true "AutoML best result")


One of the most useful tool running this experiment is the RunDetails widget where we can get different graphical elements related to the results, in the next picture we can observe a 2D graphical of the result during their execution:

![automl_02.png](./img/automl_02.png?raw=true "2D graphical")

One fact to be taken in count is we can get a slight different results during the execution of the same experiment - no changes in the jupyter notebook- at different times in Machine Learning Studio, it might be related to technical reasons such as  CPU performance, bandwith, between others, it would be great in the future to know the exact reasons. In the next picture you can see the result of VotingEnsemble was 0.8825 which is a little bit higher that 0.8822:

![automl_03.png](./img/automl_03.png?raw=true "results")

Details of the best model are shown in the next picture:

![automl_04.png](./img/automl_04.png?raw=true "best model")


In a near future I would like to expand some options in order to evaluate if the results may be improved: first I would like to try running the experiment in compute cluster with more resources (GPU instead of CPU for example), then I would like to increase the running time in order to evaluate if the results obtained are more accurated and finally I would like to expand the number of columns evaluated.

## 5. Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### 5.1 Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## 6. Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## 7. Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## 8. Standout Suggestions
1. Develop a Front End interface to interact with the deployed model, the FrontEnd will be more comprehensible for end-users
2. Suggest to the National Institute of Forensic Science of El Salvador to redesign the database of homicides, simplifying some redundant variables and recoded those ones with character only options
3. Deploy the best model found in a container, such as Kubernetes
4. Explore different algorithms to analyze the database of homicides and predict future behaviors of the phenomenon
