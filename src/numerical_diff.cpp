#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/NonLinearOptimization>
#include <vector>

#include <chrono>

using namespace std;

// sin(x0)*x1^2 - 2*x1
// 3*x1^2*x0 - cos(x0)
void NLFunc(const Eigen::VectorXd &x_in, Eigen::VectorXd &x_out)
{
    x_out[0] = sin(x_in[0]) * x_in[1] * x_in[1] - 2 * x_in[1];
    x_out[1] = 3 * x_in[1] * x_in[1] * x_in[0] - cos(x_in[0]);
}

// [cos(x0)*x1^2 | 2*sin(x0)-2 ]
// [3*x1^2 + sin(x0) | 6*x1*x0 ]
void AnalyticalJacobian(const Eigen::VectorXd& xin, Eigen::MatrixXd &jac)
{
    jac(0, 0) = std::cos(xin[0]) * xin[1] * xin[1];
    jac(0, 1) = 2 * xin[1] * std::sin(xin[0]) - 2;
    jac(1, 0) = 3 * xin[1] * xin[1] + std::sin(xin[0]);
    jac(1, 1) = 6 * xin[1] * xin[0];
}

// Generic Functor
template <typename _Scalar = double, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
    typedef _Scalar Scalar;
    enum
    {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

    int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs; }
    int values() const { return m_values; }

    int operator () (const Eigen::VectorXd& xin, Eigen::VectorXd& fval) const {
        NLFunc(xin,fval);
        return 0;
    }

};

// struct myFunctor : Functor<double>
// {
//     myFunctor() : Functor<double>(2, 2) {}
//     int operator()(const Eigen::VectorXd &xin, Eigen::VectorXd &fvec) const
//     {
//         NLFunc(xin, fvec);

//         return 0;
//     }
// };

// args @ input , nonLinear method
#define PRECISION 1e-7
void ManualNumericalJacobian(const Eigen::VectorXd &xin, Eigen::MatrixXd &jac, void (*NLFunc)(const Eigen::VectorXd &x_in, Eigen::VectorXd &x_out))
{
    Eigen::VectorXd xin_(xin);

    Eigen::VectorXd xout_(2);
    Eigen::VectorXd xout_inc(2);

    NLFunc(xin, xout_); // Nominal

    // x increment
    xin_[0] = xin[0] + PRECISION;
    xin_[1] = xin[1];

    NLFunc(xin_, xout_inc); // evaluate

    // cout << "xout_inc = " << xout_inc << endl;
    jac.col(0) = (xout_inc - xout_) / PRECISION;

    // jac(0, 0) = (xout_inc[0] - xout_[0]) / PRECISION;
    // jac(1, 0) = (xout_inc[1] - xout_[1]) / PRECISION;

    // // y increment
    xin_[0] = xin[0];
    xin_[1] = xin[1] + PRECISION;

    NLFunc(xin_, xout_inc); // evaluate

    jac.col(1) = (xout_inc - xout_) / PRECISION;

    // jac(0, 1) = (xout_inc[0] - xout_[0]) / PRECISION;
    // jac(1, 1) = (xout_inc[1] - xout_[1]) / PRECISION;
}



int main()
{
    Eigen::VectorXd xin(2);
    xin[0] = 5;
    xin[1] = -0.524;

    Functor<double> functor;
    Eigen::NumericalDiff<Functor<double>> numDiff(functor);

    Eigen::MatrixXd a_jac(2, 2);
    Eigen::MatrixXd n_jac(2, 2);
    Eigen::MatrixXd m_n_jac(2, 2);

    // Compute Jacobian
    auto start = chrono::high_resolution_clock::now();
    numDiff.df(xin, n_jac);
    auto elapsed = chrono::high_resolution_clock::now() - start;

    cout << "Eigen Diff: " << chrono::duration_cast<chrono::nanoseconds>(elapsed).count() << endl;
    // Analytical
    start =  chrono::high_resolution_clock::now();
    AnalyticalJacobian(xin, a_jac);
    elapsed =  chrono::high_resolution_clock::now() - start;
    cout << "Analytical Diff: " << chrono::duration_cast<chrono::nanoseconds>(elapsed).count() << endl;

    start =  chrono::high_resolution_clock::now();
    ManualNumericalJacobian(xin, m_n_jac, NLFunc);
    elapsed =  chrono::high_resolution_clock::now() - start;

    cout << "Manual Numerical Diff: " << chrono::duration_cast<chrono::nanoseconds>(elapsed).count() << endl;

    cout << "Jacobians ..." << endl << endl;
    cout << "Analytical Jacobian: " << a_jac << endl;
    cout << "Eigen Numerical Jacobian: " << n_jac << endl;
    cout << "Manual Numerical Jacobian: " << m_n_jac << endl;

    return 0;
}