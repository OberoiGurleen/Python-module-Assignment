#importing libraries
import csv
import statistics
import numpy as np
import random


#function read csv file and return list of lists(matrix)
def load_from_csv(file_name):
    result = np.array(list(csv.reader(open(file_name, "r"), delimiter=","))).astype("float")    #reading csv file as numpy array
    matrix = result.tolist()       #converting numpy array into list of lists
    return matrix



#function have two parameters i.e lists and return euclidean distance
def get_distance(list_1,list_2):
    point_a = np.array(list_1)
    point_b = np.array(list_2)
    return np.linalg.norm(point_a - point_b)  # np.linalg.norm(a-b) is used to calculate euclidean distance between points.



#function have two parameters a matrix and a col  no. returns standard deviation of the elements in the col no. passed
def get_standard_deviation(matrix,col_no):
    col = [i[col_no] for i in matrix]             # using list comprehension iterating through all the rows and selectively gathering all the elements occurring at col_no index.
    return statistics.stdev(col)



#function have two parameters matrix and col no. returns average of elements in the col no. passed
def get_avg(matrix,col_no):                     # get_avg function is used to get avg of a column.
    col = [i[col_no] for i in matrix]
    avg = sum(col)/len(col)
    return avg
    


#function have one parameter matrix returns standardised version of matrix
def get_standardised_matrix(matrix):
    for i,element in zip(range(len(matrix)),matrix):         #loops used to pick one value in matrix running that value through formula to get standardised matrix.
        for j in range(len(element)):
            matrix[i][j] = (matrix[i][j] - get_avg(matrix,j)) / (get_standard_deviation(matrix,j))   
    return matrix            


#function have four parameters. function finds rows of matrix_data that are the closest to the list passed as a parameter. it returns matrix solely containing labels.
def get_k_nearest_labels(alist,matrix_data,matrix_labels,k):
    distance=[]
    index=[]
    matrix=[]
    for alist_2 in matrix_data:                      #calculating distances between two lists                  
        distance.append(get_distance(alist,alist_2))                 
    dist = sorted(distance)                         #sorting the distances(ascending order)
    
    
    for i in range(k):                              #getting the index of rows of smallest distances
        for j in range(len(distance)):
            if dist[i] ==  distance[j]:
                index.append(j)

                
    for element in index:                           #finding the labels which correspond to index of rows of  smallest distances
        for j in range(len(matrix_labels)):
            if element == j:
                matrix.append(matrix_labels[j])
    return matrix


#function has one parameter matrix returns mode of matrix. if more than one number has highest frequency then function return value at random
def get_mode(matrix):
    lst=[]
    for i in matrix:
        for k in i:
            lst.append(k)
    frequency={}
    for number in lst:
        frequency.setdefault(number,0)       #default checks if the number is in our dictionary, if not then adds to our dictionary and set it to zero.
        frequency[number]+=1
    highestfrequency = max(frequency.values())  # max is used to get highest value
    highestfreqlst = []
    for number,freq in frequency.items():       # .items() is a built in python function that turns a dictionary into a list of tuples that represent key value pairs.
        if freq == highestfrequency:
            highestfreqlst.append(number)
    mode = random.choice(highestfreqlst)        # random.choice chooses a number at random
    return mode


#function have two parameters returns accuracy
def get_accuracy(correct_data_labels,data_labels):
    add=0
    for i,j in zip(correct_data_labels,data_labels):    
        if i == j:
            add+=1
    return (add/len(correct_data_labels))*100           # accuracy formula


#function have four parameters and returns matrix data_labels
def classify(data,learning_data,learning_data_labels,k):
    data_labels=[]
    dat = []
    i = 0
    for alist in data:
        labels = get_k_nearest_labels(alist,learning_data,learning_data_labels,k)
        dat.append(get_mode(labels))
    
    while i<len(dat):                          # loop used to convert list into list of lists
        data_labels.append(dat[i:i+1])
        i+=1
    return data_labels


#function is used to run series of tests
def run_test():
    #reading files and converting into list of lists
    learning_data = load_from_csv("C:\\Users\\obero\\Downloads\\File for Assessment (202021)-20201218\\Learning_Data.csv")
    learning_data_labels= load_from_csv("C:\\Users\\obero\\Downloads\\File for Assessment (202021)-20201218\\Learning_Data_Labels.csv")
    correct_data_labels = load_from_csv("C:\\Users\\obero\\Downloads\\File for Assessment (202021)-20201218\\Correct_Data_Labels.csv")
    data = load_from_csv("C:\\Users\\obero\\Downloads\\File for Assessment (202021)-20201218\\Data.csv")
    
    # running program for different k values from 3 to 15(inclusive)
    for k in range(3,16):   
        data_labels = classify(get_standardised_matrix(data),get_standardised_matrix(learning_data),learning_data_labels,k)
        accuracy = get_accuracy(correct_data_labels,data_labels)
        print('k = '+ str(k),'Accuracy = '+str(accuracy)+'%')
        
        
    
    
    
    
        
        
        
    
    
    

    
