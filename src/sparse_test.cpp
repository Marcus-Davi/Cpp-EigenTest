#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include <iomanip>
#include <chrono>

using namespace std::chrono;
using namespace Eigen;
using namespace std;

int main(int argc, char **argv)
{

    if (argc < 3)
    {
        std::cerr << "Error" << std::endl;
        exit(-1);
    }

    int n = atoi(argv[1]);
    int n2 = atoi(argv[2]);
    if (n2 > n)
    {
        std::cerr << "Error" << std::endl;
        exit(-1);
    }

    MatrixXf A = MatrixXf::Zero(n, n);
    // SparseMatrixBase<double> A_sparse;
    SparseMatrix<float> A_sparse(n, n);

    Eigen::VectorXf b(n);
    for (int i = 0; i < n; ++i)
    {
        A(i, i) = (float)std::rand() / (float)RAND_MAX;
        A_sparse.coeffRef(i, i) = A(i, i);
        b(i) = (float)std::rand() / (float)RAND_MAX;
    }

    for (int i = 0; i < n2; ++i)
    {
        int n_ = std::rand() % n;
        //std::cout << "rand index = " << n_ << std::endl;
        A(i, n_) = (float)std::rand() / (float)RAND_MAX;
        A(n_, i) = (float)std::rand() / (float)RAND_MAX;
        A_sparse.coeffRef(i, n_) = A(i, n_);
        A_sparse.coeffRef(n_, i) = A(n_, i);
    }

    int count_zeros = 0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (A(i, j) == 0)
            {
                count_zeros++;
            }
        }
    }

    // cout << A << endl << endl;
    // cout << A_sparse << endl << endl;

    //std::cout <<  std::setprecision(2) << A << std::endl;
    //
    //
    std::cout << "N# zeros: " << count_zeros << std::endl;

    //std::cout << A << b << std::endl << std::endl << res << std::endl;

    std::cout << "Using SparseLU... " << std::endl;
    auto start = high_resolution_clock::now();
    SparseLU<SparseMatrix<float>> solver;
    solver.compute(A_sparse);
    auto res2 = solver.solve(b);
    auto elapsed = duration_cast<microseconds>(high_resolution_clock::now() - start).count();
    std::cout << "Done: " << elapsed << " us" << std::endl;

    std::cout << "Using Conj.Grad ... " << std::endl;
    start = high_resolution_clock::now();
    ConjugateGradient<SparseMatrix<float>> solver2;
    solver.compute(A_sparse);
    auto res3 = solver.solve(b);
    elapsed = duration_cast<microseconds>(high_resolution_clock::now() - start).count();
    std::cout << "Done: " << elapsed << " us" << std::endl;

    std::cout << "Conventional Solving ..." << std::endl;
    start = high_resolution_clock::now();
    MatrixXf res = A.colPivHouseholderQr().solve(b);
    elapsed = duration_cast<microseconds>(high_resolution_clock::now() - start).count();
    std::cout << "Done: " << elapsed << " us" << std::endl;

    if (res.isApprox(res2))
    {
        cout << "SparseLU OK!" << endl;
    }
    else
    {
        cout << "Whoops!" << endl;
    }

    if (res.isApprox(res3))
    {
        cout << "Conj.Grad OK!" << endl;
    }
    else
    {
        cout << "Whoops!" << endl;
    }

    //		std::cout << std::setprecision(3) << A.inverse() <<std::endl;

    return 0;
}