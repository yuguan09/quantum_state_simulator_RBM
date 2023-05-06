import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# S = (vx, vy)
def dedt(S, t, g, m, b):
    vx = S[0]
    vy = S[1]
    return [
        -b/m * np.sqrt(vx**2 + vy**2) * vx, #dvx/dt
        -g - b/m * np.sqrt(vx ** 2 + vy ** 2) * vy #dvy/dt
    ]

if __name__=='__main__':
    t = np.linspace(0, 20, 100)
    m = 80
    g = 9.81
    vt = -55
    b = m*g/vt**2
    v0x, v0y = 50, 0
    sol = odeint(dedt, y0=[v0x, v0y], t=t, args=(g, m, b))

    vx = sol.T[0]
    vy = sol.T[1]

    plt.plot(t, vx)
    