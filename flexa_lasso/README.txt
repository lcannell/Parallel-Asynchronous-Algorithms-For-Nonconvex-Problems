FLEXA[1] for the solution of the LASSO problem: find x which minimizes ||Ax-b||_2^2 + lambda*||x||_1; where: lambda > 0, A has m rows and n columns.
[1] Facchinei, Scutari, Sagratella "Parallel Selective Algorithms for Nonconvex Big Data Optimization", IEEE TRANSACTIONS ON SIGNAL PROCESSING, VOL. 63, NO. 7, APRIL 1, 2015.

In order to run the code it is necessary the presence of a folder named "data" at the same level of the "main.cpp" file.
The folder must contains two files: "A.dat", "b.dat".
They define the elements of the LASSO problem and their structure must be the following.
"A.dat" must contain in the first row two numbers: m and n. Then, it will contain all the entries of the matrix A, stacked columnwise, one for each row of the file.
"b.dat" must contain the entries of the b vector of the LASSO problem, one per each row of the file.

The parameters that the user can tune are highlighted in the code as "Input parameters".