/**
 * Minimal CVODES adjoint test
 * Tests if CVodeF works with user_data like dFUSE
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>

#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_types.h>

// SUNDIALS 7.x compatibility
#if SUNDIALS_VERSION_MAJOR >= 7
using realtype = sunrealtype;
#endif

#ifndef SUN_COMM_NULL
#define SUN_COMM_NULL 0
#endif

// Test with 29 equations like dFUSE (9 soil states + 20 SWE bands)
const int NEQ = 29;
const int MAX_BANDS = 30;
const int N_TIMESTEPS = 4383;

// Mimic dFUSE's AdjointUserData structure
struct TestUserData {
    const double* forcing_flat;
    int n_timesteps;
    double params[30];
    int config_arr[10];
    double band_props[60];  // MAX_BANDS * 2
    int n_bands;
    double ref_elev;
    int n_states;
    const double* grad_output;
    std::vector<double> uh_weights;
    std::vector<double> runoff_history;
    SUNContext sunctx;
};

// RHS that uses user_data like dFUSE
static int rhs_with_userdata(realtype t, N_Vector y, N_Vector ydot, void* user_data) {
    if (!user_data || !y || !ydot) {
        std::cerr << "NULL pointer in RHS!" << std::endl;
        return -1;
    }
    
    TestUserData* data = static_cast<TestUserData*>(user_data);
    
    realtype* y_arr = N_VGetArrayPointer(y);
    realtype* ydot_arr = N_VGetArrayPointer(ydot);
    
    if (!y_arr || !ydot_arr) {
        std::cerr << "NULL array pointer in RHS!" << std::endl;
        return -1;
    }
    
    // Access user_data like dFUSE does
    int t_idx = static_cast<int>(t);
    if (t_idx >= data->n_timesteps) t_idx = data->n_timesteps - 1;
    if (t_idx < 0) t_idx = 0;
    
    // Access forcing array (like dFUSE)
    double precip = data->forcing_flat[t_idx * 3 + 0];
    double pet = data->forcing_flat[t_idx * 3 + 1];
    double temp = data->forcing_flat[t_idx * 3 + 2];
    
    // Access params (like dFUSE)
    double decay_rate = data->params[0];
    
    // Simple decay dynamics
    for (int i = 0; i < data->n_states; i++) {
        ydot_arr[i] = -decay_rate * y_arr[i] * (i + 1) * 0.001;
    }
    
    return 0;
}

int main() {
    std::cout << "SUNDIALS Version: " << SUNDIALS_VERSION_MAJOR << "." 
              << SUNDIALS_VERSION_MINOR << "." << SUNDIALS_VERSION_PATCH << std::endl;
    std::cout << "Testing with NEQ=" << NEQ << " and user_data like dFUSE" << std::endl;
    
    // Create forcing data (like dFUSE)
    std::vector<double> forcing(N_TIMESTEPS * 3);
    for (int t = 0; t < N_TIMESTEPS; t++) {
        forcing[t * 3 + 0] = 0.5;   // precip
        forcing[t * 3 + 1] = 0.0;   // pet
        forcing[t * 3 + 2] = -10.0; // temp
    }
    
    // Create grad_output (like dFUSE)
    std::vector<double> grad_output(N_TIMESTEPS, 0.0);
    
    // Setup user_data like dFUSE
    TestUserData user_data;
    std::memset(&user_data, 0, sizeof(TestUserData));
    user_data.forcing_flat = forcing.data();
    user_data.n_timesteps = N_TIMESTEPS;
    user_data.n_bands = 20;
    user_data.n_states = NEQ;
    user_data.ref_elev = 2000.0;
    user_data.grad_output = grad_output.data();
    user_data.params[0] = 0.1;  // decay rate
    user_data.uh_weights.resize(7, 0.14);
    user_data.runoff_history.resize(N_TIMESTEPS, 0.0);
    
    std::cout << "User data setup complete" << std::endl;
    
    // Create context (SUNDIALS >= 6)
    SUNContext sunctx = nullptr;
    int flag = SUNContext_Create(SUN_COMM_NULL, &sunctx);
    if (flag != 0) {
        std::cerr << "SUNContext_Create failed: " << flag << std::endl;
        return 1;
    }
    user_data.sunctx = sunctx;
    std::cout << "SUNContext created" << std::endl;
    
    // Create CVODES memory
    void* cvode_mem = CVodeCreate(CV_BDF, sunctx);
    if (!cvode_mem) {
        std::cerr << "CVodeCreate failed" << std::endl;
        return 1;
    }
    std::cout << "CVodeCreate succeeded" << std::endl;
    
    // Create vector
    N_Vector y = N_VNew_Serial(NEQ, sunctx);
    if (!y) {
        std::cerr << "N_VNew_Serial failed" << std::endl;
        return 1;
    }
    // Set initial conditions like dFUSE
    realtype* y_arr = N_VGetArrayPointer(y);
    y_arr[0] = 50.0;   // S1
    y_arr[1] = 0.0;
    y_arr[2] = 0.0;
    y_arr[3] = 0.0;
    y_arr[4] = 0.0;
    y_arr[5] = 250.0;  // S2
    for (int i = 6; i < NEQ; i++) {
        y_arr[i] = 0.0;  // SWE bands
    }
    std::cout << "N_Vector created, y[0]=" << y_arr[0] << " y[5]=" << y_arr[5] << std::endl;
    
    // Initialize CVode with user_data RHS
    flag = CVodeInit(cvode_mem, rhs_with_userdata, 0.0, y);
    if (flag != CV_SUCCESS) {
        std::cerr << "CVodeInit failed: " << flag << std::endl;
        return 1;
    }
    std::cout << "CVodeInit succeeded" << std::endl;
    
    // Set user data - CRITICAL!
    flag = CVodeSetUserData(cvode_mem, &user_data);
    if (flag != CV_SUCCESS) {
        std::cerr << "CVodeSetUserData failed: " << flag << std::endl;
        return 1;
    }
    std::cout << "CVodeSetUserData succeeded" << std::endl;
    
    // Set tolerances like dFUSE
    flag = CVodeSStolerances(cvode_mem, 1e-5, 1e-8);
    CVodeSetMaxNumSteps(cvode_mem, 10000);
    std::cout << "Tolerances set" << std::endl;
    
    // Create matrix and linear solver
    SUNMatrix A = SUNDenseMatrix(NEQ, NEQ, sunctx);
    SUNLinearSolver LS = SUNLinSol_Dense(y, A, sunctx);
    flag = CVodeSetLinearSolver(cvode_mem, LS, A);
    std::cout << "Linear solver set" << std::endl;
    
    // Test regular CVode first
    std::cout << "\n=== Testing regular CVode with user_data ===" << std::endl;
    realtype t_ret;
    for (int i = 0; i < 5; i++) {
        flag = CVode(cvode_mem, (i+1) * 1.0, y, &t_ret, CV_NORMAL);
        std::cout << "CVode t=" << t_ret << " y[0]=" << y_arr[0] << " flag=" << flag << std::endl;
        if (flag < 0) {
            std::cerr << "CVode failed!" << std::endl;
            return 1;
        }
    }
    std::cout << "Regular CVode with user_data works!" << std::endl;
    
    // Reset for adjoint test
    y_arr[0] = 50.0;
    y_arr[5] = 250.0;
    for (int i = 6; i < NEQ; i++) y_arr[i] = 0.0;
    CVodeReInit(cvode_mem, 0.0, y);
    std::cout << "\n=== Testing CVodeF with user_data ===" << std::endl;
    
    // Initialize adjoint module
    int ncheck = 0;
    int Nd = 1;
    std::cout << "Calling CVodeAdjInit with Nd=" << Nd << "..." << std::endl;
    flag = CVodeAdjInit(cvode_mem, Nd, CV_POLYNOMIAL);
    if (flag != CV_SUCCESS) {
        std::cerr << "CVodeAdjInit failed: " << flag << std::endl;
        return 1;
    }
    std::cout << "CVodeAdjInit succeeded" << std::endl;
    
    // Test CVodeF like dFUSE does - integrate to time t=1, t=2, etc.
    std::cout << "About to call CVodeF..." << std::endl;
    for (int t = 0; t < 20; t++) {
        realtype tout = static_cast<realtype>(t + 1);
        if (t < 5 || t % 5 == 0) {
            std::cout << "  CVodeF tout=" << tout << "..." << std::flush;
        }
        flag = CVodeF(cvode_mem, tout, y, &t_ret, CV_NORMAL, &ncheck);
        if (t < 5 || t % 5 == 0) {
            std::cout << " y[0]=" << y_arr[0] << " flag=" << flag << std::endl;
        }
        if (flag < 0) {
            std::cerr << "CVodeF failed at t=" << t << "!" << std::endl;
            return 1;
        }
    }
    std::cout << "CVodeF with user_data works!" << std::endl;
    
    // Cleanup
    N_VDestroy(y);
    SUNLinSolFree(LS);
    SUNMatDestroy(A);
    CVodeFree(&cvode_mem);
    SUNContext_Free(&sunctx);
    
    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
