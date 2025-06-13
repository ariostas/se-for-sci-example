from __future__ import annotations

import abc

import numpy as np


class IntegratorBase(abc.ABC):
    @abc.abstractmethod
    def compute_step(self, f, t_n, y_n, h):
        pass

    def integrate(self, f, t, init_y):
        r"""Some description of the method.

        Some more description of the method.

        .. math::
           \int f(x) dx

        Parameters
        ----------
        f : callable
            The function to integrate. It must take two arguments: the current time and the current position.
        t : array_like
            The time steps at which to compute the solution.
        init_y : float
            The initial position.

        Returns
        -------
        y : ndarray
            The solution at each time step.

        Examples
        --------
        >>> from integrator import EulerIntegrator
        >>> def f(t, y):
        ...     return -y
        >>> t = np.linspace(0, 1, 11)
        >>> init_y = 1
        >>> integrator = EulerIntegrator()
        >>> integrator.integrate(f, t, init_y)
        """
        steps = len(t)
        order = len(init_y)  # Number of equations

        y = np.empty((steps, order))
        y[0] = init_y  # Note that this sets the elements of the first row

        for n in range(steps - 1):
            h = t[n + 1] - t[n]
            y[n + 1] = self.compute_step(f, t[n], y[n], h)

        return y


class EulerIntegrator(IntegratorBase):
    def compute_step(self, f, t_n, y_n, h):
        # Compute dydt based on *current* position
        dydt = f(t_n, y_n)

        # Return next velocity and position
        return y_n - dydt


class RK4Integrator(IntegratorBase):
    def compute_step(self, f, t_n, y_n, h):
        # Compute k1 through k4
        k1 = h * f(t_n, y_n)
        k2 = h * f(t_n + h / 2, y_n + k1 / 2)
        k3 = h * f(t_n + h / 2, y_n + k2 / 2)
        k4 = h * f(t_n + h, y_n + k3)

        # Return next velocity and position
        return 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # some comment
