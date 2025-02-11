
# Assignment: Automatic Cancer Diagnostic
It is quite expensive to determine whether a particular patient has cancer or not. With this in mind,
your local NHS trust hired a software developer to design and develop a software capable of
automatically making this diagnostic. This software will read CSV files with measurements taken from
blood tests of patients and produce a diagnostic.
Unfortunately, for unknown reasons this software developer had to leave mid-project. The NHS trust
has now hired you to complete the project.
# Technical details:
You’ll be using different data structures to accomplish the below. In the text any mention to matrix
should be read as a list of lists. Your assignment must contain the code for the functions below (you
may wish to read the classification algorithm in the appendix first):
# load_from_csv
This function should have one parameter, a file name (including, if necessary, its path). The function
should read this CSV file and return a matrix (list of lists) in which a row in the file is a row in the
matrix. If the file has n rows, the matrix should have n elements (each element is a list). Notice that in
CSV files a comma separates columns (CSV = comma separated values). You can assume the file
will contain solely numeric values (and commas, of course) with no quotes.
# get_distance
This function should have two parameters, both of them lists. It should return the Euclidean distance
between the two lists. For details about this distance, read the appendix.
get_standard_deviation
This function should have two parameters, a matrix and a column number. It should return the
standard deviation of the elements in the column number passed as a parameter. For details about
how to calculate this standard deviation, read the appendix.
# get_standardised_matrix
This function should take one parameter, a matrix (list of lists). It should return a matrix containing the
standardised version of the matrix passed as a parameter. This function should somehow use the
get_standard_deviation function above. For details on how to standardise a matrix, read the
appendix.
# get_k_nearest_labels
This function should have four parameters: a list (a row of the matrix containing the data of the file
data), a matrix (list of lists) containing the data from the file learning_data, and a matrix containing the
data of the file learning_data_labels, and the last parameter is a positive integer k.
This function should find the k rows of the matrix learning_data that are the closest to the list passed
as a parameter. It should somehow use the get_distance function to do so. After finding these k rows,
it should find and return the related rows in the matrix learning_data_labels.
For example: if k=3, and the function finds that the closest rows to the list passed as parameter are
the rows 13, 26, and 34 from the matrix learning_data, then it should return a matrix (list of lists)
containing solely the rows 13, 26, and 34 from the matrix of “learning_data_labels”.
# get_mode
This function should have one parameter, a matrix (list of lists). This matrix will have only one column
(which will be returned by get_k_nearest_labels). This function should return the mode of the
numbers in this matrix. The mode of a sequence of numbers is the number with the highest frequency
(the one which repeats the most).
The mode of the sequence 5, 5, 4, 5, 4 would be 5.
If more than one number has the highest frequency, then you should return one of the numbers with
the highest frequency at random.
In the sequence 5, 5, 4, 5, 4, 1, 4 the numbers 4 and 5 have the highest frequency. You should return
either 4 or 5 at random.
5
# classify
This should have 4 parameters:
- The matrixes for data, learning_data, and learning_data_labels. For the algorithm to work correctly
the matrixes data and learning_data should be standardised beforehand.
- k, a positive integer
This function follow the algorithm described in the appendix. It should return a matrix; in this
document we call this matrix data_labels.
This function should use the other functions you wrote as much as possible. Do not keep repeating
code you already wrote.
# get_accuracy
This function should have two parameters. A matrix containing the data from the file
correct_data_labels, and a matrix containing the matrix data_labels (the output of the function
classify)
This function should calculate and return the percentage of accuracy. If both matrixes have exactly
the same values (in exactly the same row numbers) then the accuracy is of 100%. If only half of the
values of both tables match exactly in terms of value and row number, then the accuracy is of 50%,
etc.
# run_test
The aim of this function is just to run a series of tests. By consequence, here you can use hard-coded
values for the strings containing the filenames of data, correct_data_labels, learning_data and
learning_data_labels.
This function should create one matrix for each of these: correct_data_labels, learning_data and
learning_data_labels (using load_from_csv). It should standardise the matrix data and the matrix
learning_data (using get_standardised_matrix). Then, it should run the algorithm (using classify) and
calculate the accuracy (using get_acuracy) for a series of experiments. In each experiment you
should run the algorithm (and calculate the accuracy) for different values of k (go from 3 to 15 in steps
of 1), and show the results on the screen. For instance, if with k = 3 the accuracy is 68.5% it should
show:
k=3, Accuracy = 68.5%
Again, it should do the above for the values of k from 3 to 15 (inclusive), in steps of 1.
