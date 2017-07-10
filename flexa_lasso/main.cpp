/*FLEXA[1] for the solution of the LASSO problem: 

find x which minimizes ||Ax-b||_2^2 + lambda*||x||_1;

with: lambda > 0, A has m rows and n columns.

Code written by Loris Cannelli - lcannell@purdue.edu. 
Last change 07/10/2017

[1] Facchinei, Scutari, Sagratella "Parallel Selective Algorithms for Nonconvex Big Data Optimization",
IEEE TRANSACTIONS ON SIGNAL PROCESSING, VOL. 63, NO. 7, APRIL 1, 2015.
 */

#include <mpi.h>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "mkl.h"

//Functions
void compute_F(int m, int p, double *A, double *x, double *F, double *b) {	
	double *Ax = new double[m]();
	cblas_dgemv (CblasColMajor, CblasNoTrans, m, p, 1, A, m, x, 1, 0, Ax, 1);
	MPI_Allreduce(Ax, F, m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
	for(int i = 0; i < m; i++)
	    F[i] -= b[i];
	delete [] Ax;		
}

void compute_function_gradient(int m, int p, double *A, double *F, double *gradient) {	
	cblas_dgemv (CblasColMajor, CblasTrans, m, p, 1, A, m, F, 1, 0, gradient, 1);
}

double compute_function_value(int m, int p, double lambda, double *F, double *x) {	
	double f_value = 0.0, g_global = 0.0, g_local = 0.0, f = 0.0; 
	for (int i = 0; i < m; i++) 
	    f += F[i]*F[i]; 
	for(int i = 0; i < p; i++) 
	    g_local += std::abs(x[i]);
	MPI_Allreduce(&g_local, &g_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
	f_value = 0.5*f + lambda*g_global;	
	return f_value;
}

void compute_new_point(int i, double prox, double lambda, double step, double &m_value_loc, double *x, double *gradient, double *hessian) {	
    double x_opt = 0.0, parameter = 0.0, dist = 0.0;
	x_opt = x[i] - gradient[i]/(hessian[i] + prox);	
	parameter = lambda/(hessian[i] + prox);	
	if (x_opt >= parameter) 
		x_opt -= parameter;
	else {
		if (x_opt <= -parameter) 
			x_opt += parameter; 
		else 
			x_opt = 0; 
		}
	dist = x_opt - x[i];	
	if (std::abs(dist) >= m_value_loc)
	    m_value_loc = std::abs(dist);	
	x[i] += step*dist;
}

//Main program
int main (int argc, char **argv) {	
	//MPI environment initialization
	MPI::Init(argc,argv);
	int rank = MPI::COMM_WORLD.Get_rank();
	//Number of cores
	int size = MPI::COMM_WORLD.Get_size();
	int number_of_realizations = 5;	
	for(int realization = 1; realization <= number_of_realizations; realization++){		
		//Input parameters
    	int time_limit = 1000;
    	int iterations_limit = 1000;
    	int tau_decrease_l2 = 10; 
    	int max_tau_updates = 100;
    	double merit_limit = 1e-7;
    	double lip_tol = 1e-6;
    	double lambda = 1;
    	double step = 1.0;
    	double tau_step = 0.5;
    	double tau_decrease_l1 = 1e-2;
    	//Global variables and parameters
		int m = 0, n = 0, p = 0, column = 0, position = 0, iter = 0, time_flag = 0, iter_flag = 0, merit_flag = 0, function_decrements = 0, tau_updates = 0, zeros_loc = 0, zeros_global = 0;
		double entry = 0.0, f_value = 0.0, f_value_old = 0.0, m_value = 0.0, m_value_loc = 0.0, hds_local = 0.0, hds_global = 0.0, prox =0.0;
		double total_clock = 0.0, init_time = 0.0, starting_clock = 0.0, ending_clock = 0.0, elapsed_time = 0.0;
		double *times = new double[iterations_limit+2*size+1](); 
		double *values = new double[iterations_limit+2*size+1]();
		double *merits = new double[iterations_limit+2*size+1]();
		//Reading data
		if (rank == 0)
	    	std::cout<<"Reading data starts."<<std::endl; 
	    std::ostringstream instance_stream; 
		instance_stream << realization;		
		//Reading problem dimensions
		std::string data;
		data = "data/"+instance_stream.str()+"/A.dat";
		FILE *file = fopen(data.c_str(), "r");
		fscanf(file, "%lf", &entry); 
		//Number of rows of A
		m = entry;										
		fscanf(file, "%lf", &entry);
		//Number of variables 
		n = entry;			
		//Number of variables assigned to each core							
		p = ceil((double)n/size);
		//Reading matrix A entries
		double *A = new double[m*p]();														
		fscanf(file, "\n");
		for (int i = 0; i < m*n; i++) {
			column = floor(i/m); 
			fscanf(file, "%lf", &entry);
			if (p*(rank+1) > column && column >= p*rank) { 
		    	A[position] = entry;
				position++;
			}
			fscanf(file, "\n"); 
		}
		fclose(file);		
		//Reading b vector entries
		data = "data/"+instance_stream.str()+"/b.dat";
		file = fopen(data.c_str(), "r");
		double *b = new double[m]();
		for (int i = 0; i < m; i++) {
			fscanf(file, "%lf", &entry); 
			b[i] = entry;
			fscanf(file, "\n");
		}
		fclose(file);	
		if (rank == 0)	
	    	std::cout<<"Reading data ends."<<std::endl;    	
    	//Zero vector as starting point
		double *x = new double[p]();
		double *x_old = new double[p]();
		//F = Ax-b														
		double *F = new double[m]();
		double *gradient = new double[p]();			
		double *hessian = new double[p](); 																		
		//Initial computations
		double *Ax = new double[m](); 
		double f =0.0; 
		for (int i = 0; i < m; i++) 
	    	f += b[i]*b[i]; 
		//Initial objective function value
		f_value = 0.5*f;	
		delete [] Ax;				
		total_clock = MPI_Wtime();
		//Computing Hessian diagonal
		for (int i = 0; i < m*p; i++) { 
	    	int j = floor(i/m);
			hds_local += A[i]*A[i];
			hessian[j] += A[i]*A[i]; 
		}
		MPI_Barrier(MPI_COMM_WORLD); 
		MPI_Allreduce(&hds_local, &hds_global,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		//Proximal term (see [1] for details)
		prox = 0.5*std::max(std::min(1.0*hds_global/n,1.0*n),1e-10); 
		//Initial gradient
		for(int i = 0; i < m; i++)
	    	F[i] -= b[i];  
		cblas_dgemv (CblasColMajor, CblasTrans, m, p, 1, A, m, F, 1, 0, gradient, 1);	
		init_time = MPI_Wtime() - total_clock;
		//Initial merit value
		double x_opt = 0.0, parameter = 0.0, dist = 0.0;
    	for (int i = 0; i < p; i++){
	    	x_opt = x[i] - gradient[i]/(hessian[i] + prox);	
	    	parameter = lambda/(hessian[i] + prox);	
	    	if (x_opt >= parameter) 
		    	x_opt -= parameter;
	    	else {
		    	if (x_opt <= -parameter) 
			    	x_opt += parameter; 
			else 
				x_opt = 0; 
			}
	    	dist = x_opt - x[i];
	    	if (std::abs(dist) >= m_value_loc)
	        	m_value_loc = std::abs(dist);	
		}								
		MPI_Allreduce(&m_value_loc, &m_value, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);	
		if (rank == 0) {
			std::cout<<"Number of threads: "<<size<<std::endl;
			std::cout<<"Initial Value: "<<std::scientific<<std::setprecision(16)<<f_value<<std::endl; 
			std::cout<<"Initial Merit: "<< std::scientific<<std::setprecision(16)<<m_value<< std::endl;
		}				
		if (rank == 0) {
			times[iter] = 0; 
			values[iter] = f_value;
			merits[iter] = m_value;
		}		
		iter++;
		starting_clock = MPI_Wtime();	
		//FLEXA[1]						
		while (iter_flag == 0 && merit_flag == 0 && time_flag == 0) {	
        	x_old[0:p:1] = x[0:p:1]; 
        	f_value_old = f_value;
        	m_value_loc = 0.0;
			for (int i = 0; i < p; i++)					
		    	compute_new_point(i, prox, lambda, step, m_value_loc, x, gradient, hessian);				
			compute_F(m, p, A, x, F, b);
			f_value = compute_function_value(m, p, lambda, F, x);
			MPI_Allreduce(&m_value_loc, &m_value, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);	
			//Stepsize update
			step *= (1.0-1e-6*step);
			//Proximal term update
	    	if (f_value >= f_value_old && (tau_updates < max_tau_updates)) {
		    	function_decrements = 0; 
				tau_updates++;
				prox /= tau_step;
				x[0:p:1] = x_old[0:p:1];
				compute_F(m, p, A, x, F, b);
			} 
			else {
		    	function_decrements++; 
				if ((m_value < tau_decrease_l1 || function_decrements >= tau_decrease_l2) && (tau_updates < max_tau_updates)) {
			    	prox *= tau_step; 
					tau_updates++;
					function_decrements = 0;
				}  
			} 
			compute_function_gradient(m, p, A, F, gradient);	  
			if (rank == 0) {	
	        	ending_clock = MPI_Wtime();
				times[iter] = times[iter-1] + (ending_clock - starting_clock); 
				values[iter] = f_value; 
				merits[iter] = m_value;
			}		
			starting_clock = MPI_Wtime();
			iter++;          
			if (rank == 0 && (MPI_Wtime() - total_clock) >= time_limit) 
		    	time_flag = 1;
			MPI_Bcast(&time_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);					
			if (rank == 0 && iter >= iterations_limit) 
		    	iter_flag = 1;
			MPI_Bcast(&iter_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);		
			if (rank == 0 && m_value <= merit_limit) 
		    	merit_flag = 1;
			MPI_Bcast(&merit_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);				
		}
		MPI_Barrier(MPI_COMM_WORLD);
		elapsed_time = MPI_Wtime() - total_clock;	
		//Calculating number of zeros in the solution	
		for (int i = 0; i < p; i++) {
	    	if (std::abs(x[i]) <= 1e-10)
		    	zeros_loc++;	     
		}				
		MPI_Allreduce(&zeros_loc, &zeros_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); 				
		//Printing final information
		if (rank == 0) {
			std::cout<<"Final Value: "<<std::scientific<<std::setprecision(16)<<values[iter-1]<<std::endl; 
			std::cout<<"Final Merit: "<<std::scientific<<std::setprecision(16)<<merits[iter-1]<<std::endl;
			std::cout<<"Elapsed time: "<<std::scientific<<std::setprecision(16)<<elapsed_time<<std::endl;
			std::cout<<"Number of iterations: "<<std::scientific<<std::setprecision(16)<<(iter-1)<< std::endl;
			std::cout<<"Initialization time: "<<std::scientific<<std::setprecision(16)<<init_time<<std::endl;
			std::cout<<"Zeros in the solution: "<<zeros_global<<std::endl;
		}
		//Saving results
		if (rank == 0){
	    	std::cout<<"Saving results: "<<std::endl; 
	    	system("mkdir results"); 
	    	std::string folder; 
			folder = "mkdir results"+instance_stream.str();
			system(folder.c_str());
			folder = "results/"+instance_stream.str()+"/";
			std::string str_1 = folder + "parameters.dat";
			std::string str_2 = folder + "times.dat";
			std::string str_3 = folder + "values.dat";
			std::string str_4 = folder + "merits.dat";
			std::ofstream ofs_1, ofs_2, ofs_3, ofs_4; 
			ofs_1.open(str_1.c_str());
			ofs_1 << "Number of rows: " << m << std::endl;
			ofs_1 << "Number of variables: " << n << std::endl;
			ofs_1 << "Regularization_parameter: " << lambda << std::endl;
			ofs_1 << "Initialization time: " << init_time << std::endl;
			ofs_1 << "Zeros in the solution: " << zeros_global << std::endl;
			ofs_1 << "Elapsed time: " << elapsed_time << std::endl;
			ofs_1 << "Number of iterations: " << iter << std::endl;
			ofs_1.close();
			ofs_2.open(str_2.c_str());
			ofs_3.open(str_3.c_str());
			ofs_4.open(str_4.c_str());
			for (int i = 0; i < iter; i++){
				ofs_2 << std::scientific << std::setprecision(16) << times[i] << std::endl;
				ofs_3 << std::scientific << std::setprecision(16) << values[i] << std::endl;
				ofs_4 << std::scientific << std::setprecision(16) << merits[i] << std::endl;	
			} 
			ofs_2.close();
			ofs_3.close();
			ofs_4.close();
			std::cout << "DONE" << std::endl;
		}
		MPI_Barrier(MPI_COMM_WORLD);		
		//Freeing memory
		if (rank == 0)
	    	std::cout << "Freeing memory: "<<std::endl; 
		delete [] A; delete [] b; delete [] x_old; delete [] x; 
		delete [] F; delete [] gradient; delete [] hessian; 
    	delete [] times; delete [] values; delete [] merits; 
    	if (rank == 0)
	    	std::cout << "DONE" << std::endl;	
		MPI::Finalize();	
    }
	return 0;
}
