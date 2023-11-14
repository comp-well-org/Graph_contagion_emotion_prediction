# Dataset Description
To protect study participants' privacy and consent and since some of the participants did not consent to sharing their data with the third-party researchers, data has been deidentified. Furthermore, data has been transformed in a way that an individual participant's data can not be identified. This process can be reversed and data can be used to reproduce to the results presented in this work or for developing new machine learning solutions through standardization of data. Please refer to the standardization process reported in the paper.
The sensing and graph data is available in the Dataset folder. 

The feature data is in the '*feature_data.csv*'. Each row indicates a sample for a given user represented by user_id. There are multiple samples for each user taken at different timestamps. The 3rd and 4th columns of this file contains the labels for happiness and stress respectively.

The Graph Data folder contains two csv files. '*call_graph.csv*' and '*sms_graph.csv*' contain call and sms graph data respectively. 
Each csv contains a pandas dataframe with two columns: us_list and A_msg/A_call. The usr_list is an array of users present in that cohort. The corresponding A_msg/A_call is the adjacency matrix of the graph for those users.

The dates and user IDs have been anonymized for privacy concerns. However, the sequential nature of the data(differences between timestamps) is preserved.

# Code Description
The code to use the proposed model is available in the Code folder. It contains 3 main notebooks which are needed to use the models. There are multiple other python scripts containing sub routines called by these main notebooks.

1- '*GEDD.ipynb*' runs the Graph Extraction for Dynamic Distribution algorithm. It processes the Graph data files and stores new graphs in Graph_Folder.

2- '*Multimodal_data_extractor.ipynb*' extracts multi-modal sensing data based on graph networks and specified sequence length and stores it in the folder 'extracted_data'. The main loop for training and testing the model requires the data in the format as produced by the extractor.

3- '*Model_training_testing.ipynb*' is the main notebook that does bootstrapping training and testing of the proposed and benchmarks models and saves the dataframe containing centrality metrics, true, and predicted labels for all users in the test set in the Results folder.
