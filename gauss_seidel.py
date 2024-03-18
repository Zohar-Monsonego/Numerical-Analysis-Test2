import numpy as np
from numpy.linalg import norm

from colors import bcolors
from matrix_utility import *
from gauss_seidel_formula import *
from bisection_method import bisection_method
from iterative_method import find_roots_iterative_method



def find_g(A):
    _L = l_mat(A)  # calculate the L matrix
    _D = diagonal_mat(A)  # calculate the D matrix
    _U = U_mat(A)  # calculate the U matrix

    sum_L_D = sum_matrices(_L, _D)  # calculate: (L+D)
    inverse_sum_L_D = inverse(sum_L_D)  # calculate: (L+D)^-1
    inverse_mult_U = matrix_multiply(inverse_sum_L_D, _U)  # calculate: (L+D)^-1 * U
    becomes_minus = mult_matrix_in_scalar(inverse_mult_U, -1)  # calculate: -(L+D)^-1 * U
    return becomes_minus


def gauss_seidel(A, b, X0, TOL=1e-16, N=200):
    n = len(A)
    k = 1
    norm_G = norm(find_g(A))
    if norm_G > 1:
        print("The norm of G is bigger than 1 therefore the matrix will not converge with this method")
        return

    print("Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")
    x = np.zeros(n, dtype=np.double)
    while k <= N:

        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            return tuple(x)

        k += 1
        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)


if __name__ == '__main__':

    A = np.array([[2, 3, 4, 5, 6], [-5, 3, 4, -2, 3], [4, -5, -2, 2, 6], [4, 5, -1, -2, -3], [5, 5, 3, -3, 5]])
    b = np.array([70, 20, 26, -12, 37])
    X0 = np.zeros_like(b)

    solution =gauss_seidel(A, b, X0)


    #a = solution[0]
    #b = solution[1]
    #c = solution[2]
    #d = solution[3]
    #e = solution[4]


    k1 = -3
    k2 = 0
    #f = lambda x: e * x ** 3 + a * x ** 2 + b / b * x - e
    #root = bisection_method(f, k1, k2)
    print(bcolors.OKBLUE, "\nApproximate solution:", solution)


    #find_roots_iterative_method(f, k1, k2)


#print('Matrix is diagonally dominant - preforming gauss seidel algorithm\n')
