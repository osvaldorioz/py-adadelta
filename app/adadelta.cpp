#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

//c++ -O3 -Ofast -Wall -shared -std=c++20 -fPIC `python3.12 -m pybind11 --includes` adadelta.cpp -o adadelta`python3.12-config --extension-suffix`

namespace py = pybind11;

class Adadelta {
private:
    std::vector<double> weights;
    std::vector<double> accum_grad;
    std::vector<double> accum_update;
    double rho;
    double epsilon;

public:
    Adadelta(size_t num_features, double rho_val = 0.95, double eps = 1e-6)
        : weights(num_features, 0.0), accum_grad(num_features, 0.0), accum_update(num_features, 0.0), rho(rho_val), epsilon(eps) {}

    std::vector<double> get_weights() {
        return weights;
    }

    void update(py::array_t<double> gradients) {
        py::buffer_info buf = gradients.request();
        double* grad_ptr = static_cast<double*>(buf.ptr);
        
        for (size_t i = 0; i < weights.size(); ++i) {
            accum_grad[i] = rho * accum_grad[i] + (1 - rho) * grad_ptr[i] * grad_ptr[i];
            double update = - (std::sqrt(accum_update[i] + epsilon) / std::sqrt(accum_grad[i] + epsilon)) * grad_ptr[i];
            accum_update[i] = rho * accum_update[i] + (1 - rho) * update * update;
            weights[i] += update;
        }
    }
};

PYBIND11_MODULE(adadelta, m) {
    py::class_<Adadelta>(m, "Adadelta")
        .def(py::init<size_t, double, double>())
        .def("update", &Adadelta::update)
        .def("get_weights", &Adadelta::get_weights);
}
