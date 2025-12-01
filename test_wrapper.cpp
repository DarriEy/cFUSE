#include <iostream>
#include <vector>

extern "C" {
    typedef int (*external_rhs_func_t)(double t, const double* y, double* ydot, void* user_data);
    int cvodes_forward_with_checkpoints(
        external_rhs_func_t rhs_func, void* user_data, const double* y0,
        int n_states, int n_steps, double* y_out, double* runoff_out);
}

int simple_rhs(double t, const double* y, double* ydot, void* user_data) {
    int n = *static_cast<int*>(user_data);
    for (int i = 0; i < n; i++) ydot[i] = -0.001 * y[i] * (i + 1);
    return 0;
}

int main() {
    int n_states = 29, n_steps = 100;
    std::vector<double> y0(n_states, 0.0);
    y0[0] = 50.0; y0[5] = 250.0;
    std::vector<double> y_out(n_steps * n_states);
    
    std::cout << "Testing wrapper..." << std::endl;
    int result = cvodes_forward_with_checkpoints(simple_rhs, &n_states, y0.data(), n_states, n_steps, y_out.data(), nullptr);
    std::cout << "Result: " << result << std::endl;
    return result;
}
