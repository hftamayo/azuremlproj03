# Capstone Project: Homicides in El Salvador, comparing risk between men and women
### Machine Learning Engineer with Microsoft Azure Program
###### Scholarship recipient: Herbert Fernández Tamayo



#  Capstone Project: Homicides in El Salvador during 2018, comparing risk between men and women

### Table of Content
1. Project's overview
2. Project set up and installation
3. Dataset
4. Task
5. Access
6. Automated ML
7. Results
8. Hyperparameter tuning
9. Results
10. Model deployment
11. Screen recording
12. Standout suggestions

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

### Overview
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



### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
