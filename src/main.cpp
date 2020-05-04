#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
// #include <time.h>
#include <chrono> // C++ time 

int main() {

clock_t t;
std::chrono::steady_clock::time_point start,end;

int n;
std::cout << "Type Dimension of Matrix to be Inverted: ";
std::cin >> n;
MatrixXf M = MatrixXf::Random(n,n);


start = std::chrono::steady_clock::now();
MatrixXf M_i = M.inverse();
end = std::chrono::steady_clock::now();
t = clock() - t;
std::cout << "Secs  : " << std::chrono::duration_cast<std::chrono::milliseconds> (end - start).count() << " [ms]" << std::endl;

return 0;
}
