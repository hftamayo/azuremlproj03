# Capstone Project
### Machine Learning Engineer with Microsoft Azure Program
###### Scholarship recipient: Herbert Fernández Tamayo

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

- id: internal ID field for each record, this is the primary key field. [no recoded]

- regfalle: national ID number assigned for each case of homicide. [no recoded]

- edad: age by the time the person has been murdered. [no recoded]

- sexo: sex of the murdered person. Recoded values: [male -> 0, female -> 1]

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
- experiment_timeout_minutes : 20 (defines the exit criteria of each iteration)

- max_concurrent_iterations: 5 (number of thread -iteration- that can be executed simultaneously)

- primary_metric : 'accuracy' (this is the metric the experiment will try to optimize)

- task: 'classification' (key task that the experiment will focus to solve)

- label_column_name='sexo' --> key column the experiment will try to predict)


### 4.1 Results
The algorithm with the best performance during the tests was "VotingEnsemble" with a score of 0.8822, for a detailed list of the results you may check the file "automl.ipynb" section "Run Details"; in the next picture we can see the last algorithms executed and its results:

![automl_01.png](./img/automl_01.png?raw=true "AutoML best result")


One of the most useful tool running this experiment is the RunDetails widget where we can get different graphical elements related to the results, in the next picture we can observe a 2D graphical of the result during their execution:

![automl_02.png](./img/automl_02.png?raw=true "2D graphical")

One fact to be taken in count is we can get a slight different results during the execution of the same experiment - no changes in the jupyter notebook- at different times in Machine Learning Studio, it might be related to technical reasons such as  CPU performance, bandwith, between others, it would be great in the future to know the exact reasons. In the next picture you can see the result of VotingEnsemble was 0.8825 which is a little bit higher that 0.8822:

![automl_03.png](./img/automl_03.png?raw=true "results")

Details of the best model are shown in the next picture:

![automl_04.png](./img/automl_04.png?raw=true "best model")


In a near future I would like to expand some options in order to evaluate if the results may be improved: 
- Try running the experiment in compute cluster with more resources (GPU instead of CPU for example)

- Increase the running time in order to evaluate if the results obtained are more accurated and finally 

- Expand the number of columns evaluated

- Increase the number of cross validations trying to reduce the bias.


## 5. Hyperparameter Tuning
I decided to use the Logistic Regression algorithm which is part of the SciKitLearn library, to run the HyperDrive Experiment ins necessary to set the next parameters:

- C: which determines the strength of the regularization, higher values of C correspond to less regularization

- max_iter: It's the number of iteration over the full dataset.

Also, the experiment uses random parameter sampling, this one, on one hand, is very useful for discover more hyperamater combinations; on the other hand it demands more time during the execution of the experiment.

For this experiment, C was set with these values: (1, 2, 3, 4), and max_iter with (40, 80, 120, 130, 200). 

Another sets of parameters used in this experiment are:
- evaluation_interval: 1

- slack_factor: 0.2

- delay_evaluation: 5

The above parameters have relation with the accuracy of the experiment, they are useful to stop the experiment in case some conditions may be reached (that is an early termination policy), this is useful to the efficient use of compute resources.


### 5.1 Results
After the experiment was completed, the highest accuracy obtained was 0.891434, in the next picture is presented details of the results:

![hyper_05.png](./img/hyper_05.png?raw=true "best model")


In the next pictures and using the Run Details Widget, it is possible to have more details about the experiment:

- Experiment completed

![hyper_01.png](./img/hyper_01.png?raw=true "Experiment completed")

- Details of the results for each run:
![hyper_02.png](./img/hyper_02.png?raw=true "Experiment completed")

- Graphics of the result:
![hyper_03.png](./img/hyper_03.png?raw=true "Experiment completed")

![hyper_04.png](./img/hyper_04.png?raw=true "Experiment completed")


Of course, the experiment can be improved, this is a list of suggestions for future changes:
- Use Bayesian Parameter Sampling

- Use a different set of primary metrics

## 6. Model Deployment
Once we have a trained model, the next step is to deploy it using Azure Machine Learning Endpoints, to achieve this task we need:
- A trained model
- Configuration files: such as scoring and environment files
- Deploy technical details: choosing what type of container we will use (Azure Container Interfaces, Azure Kubernetes), CPU/GPU and Memory allocation

For general purpose in this project I decided to choose Azure Container Interface (ACI) with 1 CPU and 1 GB of Memory.

Even though the rubric of the project clearly specify I should deploy the best model obtained, I decided to implement both of them for the next reasons:
- I want to compare if the deployment process is the same for both models.
- Learn about pitfalls or special configuration for each model.
- The ooportunity to learn something new.

In the next paragraphs I will describe how I deployed each model:

### 6.1 Deploying AutoML Model:

In the case of the AutoML experiment, the score file was generated by the model, the purpose of this file is to give information about what type of input data the model expects to process it and return results; It is useful to have the information from the environment method of the best model obtained to deploy the webservice. In the next image shows the detail of the code of how the webservice is created and deployed:

![automlexp_wservice_scode.png](./img/automlexp_wservice_scode.png?raw=true "AutoML webservice sourcecode")

To test de deployed model in the WebService we must use data in JSON format, of course, details in the scoring.py file are important to obtain output, in the case of this experiment, the output was like this (remember 0 in this experiment resembles "male"):

![automlexp_wservice_requests_results.png](./img/automlexp_wservice_requests_results.png?raw=true "AutoML webservice results")


Details about the Endpoint are given in the next images:

![automlexp_endpoint_widget.png](./img/automlexp_endpoint_widget.png?raw=true "AutoML Endpoint widget")

![automlexp_endpoint_status.png](./img/automlexp_endpoint_status.png?raw=true "AutoML Endpoint widget")

![automlexp_endpoint_status02.png](./img/automlexp_endpoint_status02.png?raw=true "AutoML Endpoint widget")

![automlexp_endpoint_status03.png](./img/automlexp_endpoint_status03.png?raw=true "AutoML Endpoint widget")


As a good practice, in the jupyter file, at the end of it, the webservice and the compute cluster are deleted because I won't be accesible anymore, this process reminds me the garbage class collection of JAVA:

![automlexp_deleting_ccluster.png](./img/automlexp_deleting_ccluster.png?raw=true "AutoML Endpoint widget")


Some pitfalls during the deployment process:
- Be careful how your model is named and passed to the InferenceConfig.

- Most common exception: FileNotFoundException, PathNotFoundException.

- Double check your dataset, any inconsistency may affect the deployment process and the interaction with the webservice

- Understand first how you will interact with the model and how the test data has to be transformed to JSON.


### 6.2 Deploying HyperParameter Model:

The most important conclusion is deploying the hyperparameter model is not the same as the AutoML -no, copy and paste won't work- one of the aspect to be taken in count is how the model is registered, the best way I found is to obtained during the training of the experiment, that's mean, train.py is the key for this. Then, please check if the model's file is uploaded to the environment, be sure of the name of it. From here, you need to register the model using the "register_model()" method.


Another diference is I have to write the scoring file (score.py) following the guidelines from the official documentation, please check score.py from this repo for further details.

The sourcode from my hyperdrive jupyter notebook file to implement the model is this:

![hyper_wservice_scode.png](./img/hyper_wservice_scode.png?raw=true "Hyper webservice sourcecode")




About the endpoint, in the next images there are technical information that might be useful to check its healthy:


![hyperws_06.png](./img/hyperws_06.png?raw=true "Hyper webservice deployed")

![hyperws_07.png](./img/hyperws_07.png?raw=true "Hyper webservice deployed")

![hyperws_08.png](./img/hyperws_08.png?raw=true "Hyper webservice deployed")


Some pitfalls during the deployment process:
- Be sure to understand the process how your model is obtained and registered.

- Don't try to obtain the scoring and the environment files automatically, it was a waste of time at least for me because I couldn't do that.

- The output of your webservice is heavily influenced by how you coded your score.py

- Be sure how to convert your test data into JSON format.

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


## 9. Best practice:
As a part of best practices, I tried to convert my HyperDrive best model to ONNX format, this is the sourcecode and the results:
