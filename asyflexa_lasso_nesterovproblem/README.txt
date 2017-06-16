AsyFLEXA[1] for the solution of the LASSO problem: find x which minimizes ||Ax-b||_2^2 + lambda*||x||_1; where: lambda > 0, A has m rows and n columns.
[1] Cannelli, Facchinei, Kungurtsev, Scutari "Asynchronous Parallel Algorithms for Nonconvex Big-Data Optimization. Part I: Model and Convergence. arXiv preprint arXiv:1607.04818, 2016.

In order to run the code it is necessary the presence of a folder named "data" at the same level of the "main.cpp" file.
The folder must contains three files: "A.dat", "b.dat", "xs.dat".
The first two files define the elements A and b of the LASSO problem. The third one is the vector of the optimal solution, which is known for this class of LASSO problems.
The structure of the files must be the following.
"A.dat" must contain in the first row two numbers: m and n. Then, it will contain all the entries of the matrix A, stacked columnwise, one for each row of the file.
"b.dat", "xs.dat" must contain respectively the entries of the b vector of the LASSO problem and the entries of the optimal solution of the problem, one per each row of the files.

The parameters that the user can tune are highlighted in the code as "Input parameters".

Note that "compute_function_gradient" can be implemented in two equivalent ways (one approach is currently written as a comment inside the function): if the dimension of the block-variables is small, leave the code as it is
(now this dimension is set as P = 1). If it is of interest to enlarge the dimension of the block-variables, then the code will run faster by uncommenting the second way to compute the gradient.
 