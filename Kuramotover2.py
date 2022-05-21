import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import math
np.seed = 42
#parameters
alpha = math.pi/3
n = 100 #number of oscillators
ep = .001
omega = 0.5
beta = 0.0


tf = 100 #final time

#Adjacency matrix
K = np.ones((n,n), dtype=int)

def KuramotoModel(t, U):
    phi = U

    sys = np.zeros(len(U))

    for i in range(n):
        sum = 0
        for j in range(n):
            dkij = -ep*(math.sin(phi[i] - phi[j] + beta) + K[i,j])
            sum +=  K[i,j]*math.sin(phi[i] - phi[j] + alpha)
        sys[i] = omega - (1 / n) * sum
    return sys


phi_0 = np.random.uniform(0,2*math.pi,n) #initial conditions

soln = solve_ivp(KuramotoModel,t_span=(0,tf),y0 = phi_0) #solves ode

plt.close('all')
fig,ax = plt.subplots(3,1)
for i in range(n):
    ax[0].plot(soln.t, soln.y[i]%(2*np.pi))
ax[0].title.set_text("Phases")

#initial phase
ax[1].scatter(np.cos(soln.y[:,1]), np.sin(soln.y[:,1]))
ax[1].plot(np.cos(np.linspace(-np.pi,np.pi,1000)),np.sin(np.linspace(-np.pi,np.pi,1000)))
ax[1].title.set_text("Initial Phase")

#final phase
ax[2].scatter(np.cos(soln.y[:,-1]), np.sin(soln.y[:,-1]))
ax[2].plot(np.cos(np.linspace(-np.pi,np.pi,1000)),np.sin(np.linspace(-np.pi,np.pi,1000)))
ax[2].title.set_text("Final Phase")
plt.tight_layout()
plt.show()