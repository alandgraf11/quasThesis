def rk4_step_ca(f, x, u, h): # RK4 integrator for single step numerical integration
    """ one step of ode integration with Runge-Kutta of order 4
        f:  implementation of an ode, such that dx/dt = f(x, u)
            (u is assumed constant)
        x:  initial value
        h: time step
    """
    k1       = f(x, u)
    k2       = f(x + h/2 * k1, u)
    k3       = f(x + h/2 * k2, u)
    k4       = f(x + h * k3, u)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)