# Dataset Description
The sensing and graph data is available in the Dataset folder.

The feature data is in the 'feature_data.csv'. Each row indictaes a sample for a given user represented by user_id. There are multiple samples for each user taken at different timestamps. The 3rd and 4th columns of this file contains the labels for happiness and stress respectively.

The Graph Data folder contains two csv files. 'call_graph.csv' and 'sms_graph.csv' contain call and sms graph data respectively. 
Each csv contains a pandas dataframe with two columns: us_list and A_msg/A_call. The usr_list is an array of users present in that cohort. The corresponding A_msg/A_call is the adjacency matrix of the graph for those users.

# Code Description
The code to use the proposed model is available in the Code folder. It contains 3 main notebooks which are needed to use the models. There are multiple other python scripts containing sub routines called by these main notebooks.

1- '*GEDD.ipynb*' runs the Graph Extraction for Dynamic Distribution algorithm. It processes the Graph data files and stores new graphs in Graph_Folder.

2- '*Multimodal_data_extractor.ipynb*' extracts multi-modal sensing data based on graph networks and specified sequence length and stores it in the folder 'extracted_data'. The main loop for training and testing the model requires the data in the format as produced by the extractor.

3- '*Model_training_testing.ipynb*' is the main notebook that does bootstrapping training and testing of the proposed and benchmarks models and saves the dataframe containing centrality metrics, true, and predicted labels for all users in the test set in the Results folder.
