In order to run the code it is necessary the presence of a folder named "data" at the same level of the "main.cpp" file.
The folder must contains three files: "A.dat", "b.dat", "xs.dat".
The first two files define the elements of the LASSO problem. The third one is the vector of the optimal solution, which is known for this class of LASSO problems.
The structure of the files must be the following.
"A.dat" must contain in the first row two numbers: m and n, of the LASSO problem. Then, it will contain all the entries of the matrix A, stacked columnwise, one for each row of the file.
"b.dat", "xs.dat" will contain respectively the entries of the b vector of the LASSO problem and the entries of the optimal solution of the problem, one per each row of the files.

The parameters that the user can tune are highlighted in the code as "Input parameters".
