/*AsyFLEXA[1] for the solution of the Nesterov's LASSO problem[2]: 

find x which minimizes ||Ax-b||_2^2 + lambda*||x||_1;

with: lambda > 0, A has m rows and n columns.

Code written by Loris Cannelli - lcannell@purdue.edu. 
Last change 06/15/2017

[1] Cannelli, Facchinei, Kungurtsev, Scutari "Asynchronous Parallel Algorithms for Nonconvex Big-Data Optimization. Part I: Model and Convergence",
arXiv preprint arXiv:1607.04818, 2016.
[2] Nesterov "Gradient methods for minimizing composite functions",
MATHEMATICAL PROGRAMMING, 2013.
 */

#include <omp.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "mkl.h"

// Functions
void compute_function_gradient(double *x, double *gradient, double *ATA, double *ATb, int actual_index, int P, int n) {
	double *ATAx = new double[P]();
	for (int k = 0; k < n; k++) {
		for (int j = 0; j < P; j++) 
			ATAx[j] += ATA[n*actual_index + n*j + k]*x[k];     
			//ATAx[j] += ATA[actual_index + j + n*k]*x[k];  
	}
	for (int j = 0; j < P; j++) 
	    gradient[j] = ATAx[j] - ATb[actual_index + j];
	delete [] ATAx;		
}

double compute_function_value(int m, int n, double lambda, double *x, double *A, double *b) {	
	double f1 = 0.0, g = 0.0, f_value = 0.0; 
	double *Ax = new double[m]();
	for (int i = 0; i < n; i++) {
		g += std::abs(x[i]);
		for (int j = 0; j < m; j++) 
			Ax[j] += A[i*m+j]*x[i];			
	}
	for (int i = 0; i < m; i++)
	     f1 += (Ax[i]-b[i])*(Ax[i]-b[i]);
	f_value = 0.5*f1 + lambda*g;
    delete [] Ax;
	return f_value;										 								
}

//Main program
int main (void) {
    srand(time(NULL));		
	//Input Parameters 	
	int time_limit = 1000;
	int iterations_limit = 1000;
	int number_of_threads = 1;
	int P = 1; //block-variable dimension
	int tau_decrease_l2 = 10; 
	int max_tau_updates = 100; 	
	double merit_limit = 1e-7;	
	double lambda = 1.0;
    double step = 0.95;	
    double tau_step = 0.5;
    double tau_decrease_l1 = 1e-2;
	//Global variables and parameters
	int m = 0, n = 0, p = 0, N = 0, iter = 0, fe = 0, thread_id = 0, time_flag = 0, iter_flag = 0, merit_flag = 0, function_decrements = 0, tau_updates = 0, zeros = 0;
	int core_count = 0;
	double entry = 0.0, f_star = 0.0, hds_global = 0.0, prox =0.0, btb = 0.0;
	double total_clock = 0.0, init_time = 0.0, starting_clock = 0.0, ending_clock = 0.0, elapsed_time = 0.0;
	int *actual_iteration_vector = new int[number_of_threads]();
	double *times = new double[iterations_limit+2*number_of_threads+1](); 
	double *values = new double[iterations_limit+2*number_of_threads+1]();
	double *merits = new double[iterations_limit+2*number_of_threads+1]();
	//Reading data
	std::cout<<"Reading data starts."<<std::endl; 
	//Reading problem dimensions
	std::string data;
	data = "data/A.dat";
	FILE *file = fopen(data.c_str(), "r");
	fscanf(file, "%lf", &entry); 
	//Number of rows of A
	m = entry;										
	fscanf(file, "%lf", &entry);
	//Number of variables 
	n = entry;												
	double *A = new double[m*n]();														
	fscanf(file, "\n");
	for (int i = 0; i < m*n; i++){
		fscanf(file, "%lf", &entry);
		A[i] = entry;
		fscanf(file, "\n"); 
	}
	fclose(file);		
	//Reading b vector entries
	data = "data/b.dat";
	file = fopen(data.c_str(), "r");
	double *b = new double[m]();
	for (int i = 0; i < m; i++){
		fscanf(file, "%lf", &entry); 
		b[i] = entry;
		fscanf(file, "\n");
	}
	fclose(file);
	//Reading optimal solution entries
	data = "data/xs.dat";
	file = fopen(data.c_str(), "r");
	double *x_star = new double[n]();
	for (int i = 0; i < n; i++){
		fscanf(file, "%lf", &entry); 
		x_star[i] = entry;
		fscanf(file, "\n");
	}
	fclose(file);		
	std::cout<<"Reading data ends."<<std::endl;  
	//Quantity of scalar variables for each core
	p = n/number_of_threads;
	//Number of block-variables for each core
	N = p/P; 
    //Zero vector as starting point
    double x[n]; 
	std::fill(x, x+n, 0.0);
	double *ATA = new double[n*n];
	double *ATb = new double[n];		
	//Initial computations
	double f1 = 0.0, f2 = 0.0, g = 0.0, f_value_init = 0.0, m_value_init = 0.0;
	double *ATAx = new double[n]();
	total_clock = omp_get_wtime();
	cblas_dgemm (CblasColMajor, CblasTrans, CblasNoTrans, n, n, m, 1, A, m, A, m, 0, ATA, n);
	init_time = omp_get_wtime() - total_clock;
	elapsed_time = omp_get_wtime() - total_clock;
	cblas_dgemv (CblasColMajor, CblasTrans, m, n, 1, A, m, b, 1, 0, ATb, 1);
	for (int i = 0; i < m; i++) 
	    btb += b[i]*b[i];	
	//Optimal value										 													
	cblas_dgemv (CblasColMajor, CblasTrans, n, n, 1, ATA, n, x_star, 1, 0, ATAx, 1);
	for (int i = 0; i < n; i++) {
		f1 += x_star[i]*ATAx[i]; 
		f2 += x_star[i]*ATb[i];
		g += std::abs(x_star[i]);
	}
	f_star = 0.5*(f1 + btb) - f2 + lambda*g;
	//Initial objective function value
	f_value_init = 0.5*btb;
	delete [] ATAx;	
	//Initial relative error
	m_value_init = (f_value_init - f_star)/f_star;	
	std::cout<<"Number of threads: "<<number_of_threads<<std::endl;
	std::cout<<"Optimal value: "<<std::scientific<<std::setprecision(16)<<f_star<<std::endl;
	std::cout<<"Initial Value: "<<std::scientific<<std::setprecision(16)<<f_value_init<<std::endl; 
	std::cout<<"Initial Merit: "<<std::scientific<<std::setprecision(16)<<m_value_init<<std::endl;		
	values[iter] = f_value_init;
	merits[iter] = m_value_init;	
	times[iter] = 0.0;
	iter++;					
	total_clock = omp_get_wtime();	
	//Computing Hessian diagonal						
	double *hessian = new double[n](); 
	for (int i = 0; i < m*n; i++){ 
    	int j = floor(i/m); 
	    hds_global += A[i]*A[i]; 
  	    hessian[j] += A[i]*A[i]; 
	} 	
	//Proximal term (see [1] for details)
	prox = 0.5*std::max(std::min(1.0*hds_global/n,1.0*n),1e-10);
    init_time += omp_get_wtime() - total_clock;		 
	//OpenMP environment initialization  		       		
	#pragma omp parallel num_threads(number_of_threads) private(fe, thread_id) 
	{			
		thread_id = omp_get_thread_num(); 
		fe = thread_id*p; 			
		int current_index = 0, actual_index = 0, actual_iteration  = 1;
		double f_value = 0.0, m_value = 1000, f_value_old = values[0], x_opt = 0.0, parameter = 0.0;
		double starting_clock = 0.0, ending_clock = 0.0;
		int indici[N]; 
		double gradient[P];
        double *times_local = new double[iterations_limit+1](); 
        double *values_local = new double[iterations_limit+1](); 			
        double *merits_local = new double[iterations_limit+1](); 			
		times_local[0] = times[0];
		values_local[0] = values[0];
		merits_local[0] = merits[0];
		actual_iteration_vector[thread_id] = 1;
		for (int i = 0; i < N; i++) 
			indici[i] = i; 
		starting_clock = omp_get_wtime();
		//AsyFLEXA[1]			
		while (iter_flag == 0 && merit_flag == 0 && time_flag == 0) {	
			for (int i = 0; i < N; i++) {					
				actual_index = fe + indici[i]*P;
				compute_function_gradient(x, gradient, ATA, ATb, actual_index, P, n);
				for (int j = 0; j < P; j++) {
					current_index = actual_index + j;		
					x_opt = x[current_index] - gradient[j]/(hessian[current_index]+prox);	
	                parameter = lambda/(hessian[current_index]+prox);
	                if (x_opt >= parameter) 
		            	x_opt -= parameter;
	                else {
		            	if (x_opt <= -parameter) 
	                    	x_opt += parameter; 
		                else 
			            	x_opt = 0.0; 
		            }
	                x[current_index] += step*(x_opt - x[current_index]); 								
				}
			}				
			std::random_shuffle (indici, indici+N);				
			core_count++;
			if (core_count >= number_of_threads){	
				core_count = 0;					
				ending_clock = omp_get_wtime();
				f_value = compute_function_value(m, n, lambda, x, A, b);
				m_value = (f_value - f_star)/f_star;		
				times_local[actual_iteration] = times_local[actual_iteration-1]+(ending_clock-starting_clock);
				values_local[actual_iteration] = f_value; 
				merits_local[actual_iteration] = m_value;
				actual_iteration++;
				actual_iteration_vector[thread_id]++;
				starting_clock = omp_get_wtime();
				//Proximal term update
	            if ((f_value >= f_value_old) && (tau_updates < max_tau_updates)) {
			    	function_decrements = 0; 
					tau_updates++; 
					prox /= tau_step;
			    } 
		        else {
		           function_decrements++;
		           if ((m_value < tau_decrease_l1 || function_decrements >= tau_decrease_l2) && (tau_updates < max_tau_updates)) {
       	           		prox *= tau_step; 
						tau_updates++; 
						function_decrements = 0;
				    }
			    }	 
			    f_value_old = f_value;	
			}
			//Stepsize update
			step *= (1.0-1e-6*step);
			iter++;					
			if ((omp_get_wtime() - total_clock) >= time_limit)
				time_flag = 1; 
			if (m_value <= merit_limit)
			    merit_flag = 1; 
			if (iter >= iterations_limit)
			    iter_flag = 1;
		}
		ending_clock = omp_get_wtime();
		#pragma omp barrier
		if (thread_id == 0) {
		f_value = compute_function_value(m, n, lambda, x, A, b);
	    m_value = (f_value - f_star)/f_star;
		times_local[actual_iteration] = times_local[actual_iteration-1]+(ending_clock-starting_clock);
		values_local[actual_iteration] = f_value; 
		merits_local[actual_iteration] = m_value;
		actual_iteration_vector[thread_id]++;
		//Calculating number of zeros in the solution	
		for (int i = 0; i < n; i++) {
		    if (std::abs(x[i]) <= 1e-10)
			    zeros++;	    
			}
		}
		#pragma omp barrier
		//Printing final information
		int value = 0;
		for(int i = 0; i < thread_id; i++)
		    value +=  (actual_iteration_vector[i]-1);
		for (int i = 0; i < (actual_iteration_vector[thread_id]-1); i++){
		    times[value+1+i] = times_local[i+1];
			values[value+1+i] = values_local[i+1];
			merits[value+1+i] = merits_local[i+1];
		}	 
		if (thread_id == 0) {
		    std::cout<<"Final Value: "<<std::scientific<<std::setprecision(16)<<f_value<<std::endl; 
		    std::cout<<"Final Merit: "<<std::scientific<<std::setprecision(16)<<m_value<<std::endl;
		    std::cout<<"Zeros: "<<zeros<<std::endl;
		}
		#pragma omp barrier
	}
	elapsed_time += omp_get_wtime() - total_clock;				
	std::cout<<"Elapsed time: "<<std::scientific<<std::setprecision(16)<<elapsed_time<<std::endl;
	std::cout<<"Number of iterations: "<<std::scientific<<std::setprecision(16) <<iter<<std::endl;	
	//Saving results	
	std::cout<<"Saving results: "<<std::endl; 
	system("mkdir results"); 
	std::string folder; 
	folder = "mkdir results";
	system(folder.c_str());
	folder = "results/";
	std::string str_1 = folder + "parameters.dat";
	std::string str_2 = folder + "times.dat";
	std::string str_3 = folder + "values.dat";
	std::string str_4 = folder + "merits.dat";
	std::ofstream ofs_1, ofs_2, ofs_3, ofs_4; 
	ofs_1.open(str_1.c_str());
	ofs_1 << "Number of rows: " << m << std::endl;
	ofs_1 << "Number of variables: " << n << std::endl;
	ofs_1 << "Regularization_parameter: " << lambda << std::endl;
	ofs_1 << "Optimal value: " << f_star << std::endl;
	ofs_1 << "Initialization time: " << init_time << std::endl;
	ofs_1 << "Zeros in the solution: " << zeros << std::endl;
	ofs_1 << "Elapsed time: " << elapsed_time << std::endl;
	ofs_1 << "Number of iterations: " << iter << std::endl;
	ofs_1.close();
	ofs_2.open(str_2.c_str());
	ofs_3.open(str_3.c_str());
	ofs_4.open(str_4.c_str());
	for (int i = 0; i < iter; i++) {
		ofs_2 << std::scientific << std::setprecision(16) << times[i] << std::endl;
		ofs_3 << std::scientific << std::setprecision(16) << values[i] << std::endl;
		ofs_4 << std::scientific << std::setprecision(16) << merits[i] << std::endl;	
	} 
	ofs_2.close();
	ofs_3.close();
	ofs_4.close();
	std::cout<<"DONE"<<std::endl;
	//Freeing memory
	std::cout<<"Freeing memory: "<<std::endl; 
	delete [] A; delete [] b; delete [] x_star;	delete [] ATA; delete [] ATb; delete [] hessian;
	delete [] times; delete [] values; delete [] merits; delete [] actual_iteration_vector;
	std::cout<<"DONE"<< std::endl;
	return 0;
}
