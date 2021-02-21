#include <iostream>
#include <vector>
// #include <unsupported/Eigen/NumericalDiff>
#include <unsupported/Eigen/NonLinearOptimization>

#include "matplotlibcpp.h"

#include <chrono>

using namespace std;
using namespace Eigen;
namespace plt = matplotlibcpp;

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
    Functor(int values) : m_values(values) {}

    int inputs() const { return m_inputs; }
    int values() const { return m_values; }
};

struct Model : Functor<double>
{

    Model(const VectorXd &input_data, const VectorXd &output_data) : Functor<double>(input_data.size())
    {

        model_input = input_data;
        model_output = output_data;
    }

    VectorXd model_input;
    VectorXd model_output;

    //y(xi) = b0*xi/(b1 + xi)
    int operator()(const Eigen::VectorXd &xin, Eigen::VectorXd &fvec) const
    {
        // cout << "xin =  " << xin << endl;
        Model::calls_counter++;
        for (int i = 0; i < values(); ++i)
        {
            fvec[i] = model_output[i] - (xin[0] * model_input[i]) / (xin[1] + model_input[i]); // cost Function
        }

        return 0;
    }

    static void resetCallCounter() {
        calls_counter = 0;
    }

    static int getCallsCounter(){
        return calls_counter;
    }

    // private:
    static int calls_counter;
};

// Problem : Fit following data to  model [ y(xi) = b0*xi/(b1 + xi) ]
vector<double> input = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.740};
vector<double> output = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};



void EvaluateModel(const VectorXd &parameters, const VectorXd &model_input, VectorXd &output_data)
{
    int N = model_input.size();

    if (model_input.size() != output_data.size())
    {
        cerr << "Evaluation Wrong sizes" << endl;
        exit(-1);
    }

    for (int i = 0; i < N; ++i)
    {
        output_data[i] = (parameters[0] * model_input[i]) / (parameters[1] + model_input[i]); // cost Function
    }
}
int Model::calls_counter = 0;

int main()
{

    
    VectorXd input_data(input.size());
    VectorXd output_data(input.size());

    for (int i = 0; i < input.size(); ++i)
    {
        input_data[i] = input[i];
        output_data[i] = output[i];
    }

    VectorXd parameters(2);
    parameters.setZero();

    Model functor(input_data, output_data);
    NumericalDiff<Model> num_diff(functor); // Numerical Differentiation

    Model::resetCallCounter();

    auto start = chrono::high_resolution_clock::now();
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<Model>, double> lm(num_diff);
    int info = lm.minimize(parameters);
    auto elapsed = chrono::high_resolution_clock::now() - start;
    cout << "Execution time:  " << chrono::duration_cast<chrono::microseconds>(elapsed).count() << "us" << endl;
    cout << "Iterations(?): " << lm.iter << endl;
    cout << "Function Calls: " << Model::calls_counter << endl;
    // cout << "nfev: " << lm.nfev << endl; // Same as Function Calls
    // cout << lm.fjac << endl;

    // return 0;
    



    cout << "Creating Test Data..." << endl;
    VectorXd test_data(100);
    double f = 0;
    double increment = 4.0 / 100.0;
    for (int i = 0; i < test_data.size(); ++i)
    {
        test_data[i] = f;
        f += increment;
    }

    // Declaring variable that will receive data;
    VectorXd model_output(test_data.size());

    cout << "Evaluating Model..." << endl;
    // Execute the nonlinear method over input data / saves data to a given vector. Resizing is not performed internally.
    EvaluateModel(parameters, test_data, model_output);

    // It feels unwise to copy VectorXd to std::vector<> simply to plot the data.. I found no way to typecast though.
    vector<double> model_output_vec(model_output.data(), model_output.data() + model_output.size());
    vector<double> test_data_vec(test_data.data(), test_data.data() + test_data.size());

    cout << "Final Parameters: " << parameters << endl;
    plt::named_plot("Observations", input, output, "r.");
    plt::named_plot("Model Output", test_data_vec, model_output_vec, "b");
    plt::legend();
    // plt::plot(output);
    plt::show();

    return 0;
}