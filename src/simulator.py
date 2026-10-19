import numpy as np
from utils.quaternion import normalize_q

class Simulator():

    def __init__(self, config, dynamics_func, integrator):
        self.dynamics_func = dynamics_func          # self.f (rhs as a casadi function) from vehicle.ca_model
        self.integrator = integrator            
        self.dt = config['simulation']['time_step']

    def step(self, state, controls): # one RK4 step
        return self.integrator(self.dynamics_func, state, controls, self.dt)

    def integrate(self, state, controls, t0, tf): # repeated RK4 steps
        t = t0
        if state.ndim == 1:
            state = state.reshape(len(state), 1)
        states = [state]
        times = [t]

        while t < tf:
            state = self.step(state, controls)
            state = normalize_q(state)
            t += self.dt
            states.append(state)
            times.append(t)

        return np.array(times), np.array(states)