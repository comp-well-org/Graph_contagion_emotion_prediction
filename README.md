# Dataset Details
The sensing and graph data is available in the Dataset folder.

The feature data is in the 'feature_data.csv'. Each row indictaes a sample for a given user represented by user_id. There are multiple samples for each user taken at different timestamps. 

The Graph Data folder contains two csv files. 'call_graph.csv' and 'sms_graph.csv' contain call and sms graph data respectively. 
Each csv contains a pandas dataframe with two columns: us_list and A_msg/A_call. The usr_list is an array of users present in that cohort. The corresponding A_msg/A_call is the adjacency matrix of the graph for those users.

