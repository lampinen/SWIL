import numpy as np
import matplotlib.pyplot as plot
from orthogonal_matrices import random_orthogonal

############ config
num_runs = 1
num_input = 100
num_output = 1 # TODO: only partially implemented
rank = 1 # TODO: Probably impossible 
num_epochs = 5000
esses = [10, 5, 2]
init_size = 1e-4
epsilon = 0.01
lr = 0.001
#############
tau = 1./lr 

def _train(sigma_31, sigma_11, W21, W32, num_epochs):
    tracks = {
        "W21": np.zeros([num_epochs, num_input]),
        "W32": np.zeros([num_epochs, num_output]),
        "loss": np.zeros([num_epochs])
        }
    for epoch in range(num_epochs):
        l = sigma_31 - W32* np.dot(W21, sigma_11)
        W21 += lr * W32 * l 
        W32 += lr * np.dot(l, W21.transpose()) 
        tracks["W21"][epoch, :] = W21
        tracks["W32"][epoch, :] = W32
        tracks["loss"][epoch] = np.mean(np.square(l))

    return W21, W32, tracks

def _ugly_function(sc, c0, two_s, theta):
    return np.log((sc + c0 + two_s*np.tanh(theta/2.))/(sc - (c0 + two_s*np.tanh(theta/2.))))

def _estimated_learning_time(a0, b0, s, epsilon, tau):
    c0 = np.abs(a0**2 - b0**2) / 2.
    theta0 = np.arcsinh(2*a0*b0/c0)
    thetaf = np.arcsinh((1-epsilon) * 2*s/ c0)

    sc = np.sqrt(c0**2 + (2*s)**2)

    t = (tau/sc)*(_ugly_function(sc, c0, 2*s, thetaf) - _ugly_function(sc, c0, 2*s, theta0))
    return t


for run_i in range(num_runs):
    for s in esses:
        original_mode = random_orthogonal(num_input)[0, :]
        new_mode = np.copy(original_mode)
        new_mode[:10] = -original_mode[:10] # a perturbation!
        input_data = np.eye(num_input) # random_orthogonal(num_input)
        sigma_31 = s*np.dot(original_mode, input_data)
        new_sigma_31 = s*np.dot(new_mode, input_data)
        sigma_11 = np.eye(num_input)
        
        # initial weights
        W21 = np.sqrt(init_size)*random_orthogonal(num_input)[0, :]
        if np.random.randn() > 0:
            W32 = np.sqrt(init_size)
        else:
            W32 = -np.sqrt(init_size)

        # learning from random init -- theory
        a0 = np.dot(W21, original_mode)
        b0 = W32 
        est1 = _estimated_learning_time(a0, b0, s, epsilon, tau)

        # learning from random init -- empirical
        W21, W32, first_tracks = _train(sigma_31, sigma_11, W21, W32, num_epochs)   
        print(est1)
        est1_int = int(est1)
        print(np.dot(first_tracks["W21"][est1_int, :], original_mode))
        print(first_tracks["W32"][est1_int, :])
        print(s * (1-epsilon))
        
        # updating to new situation -- theory
        a0 = np.dot(W21, new_mode)
        b0 = W32 
        est2 = _estimated_learning_time(a0, b0, s, epsilon, tau)

        # updating to new situation --empirical 
        W21, W32, second_tracks = _train(new_sigma_31, sigma_11, W21, W32, num_epochs)
        print(est2)
        est2_int = int(est2)
        print(np.dot(second_tracks["W21"][est2_int, :], new_mode))
        print(second_tracks["W32"][est2_int, :])
        print(s * (1-epsilon))

        # plotting
        epochs = range(num_epochs)
        plot.figure()
        plot.plot(epochs, first_tracks["loss"])
        plot.axvline(x=est1, color='r')
        plot.xlabel("epoch")
        plot.ylabel("loss (first phase)")
        plot.legend(["Empirical", "Theoretical time of 99% learning"])
        plot.savefig("results/singular_value_%.2f_initial_learning.png" % s)
        plot.figure()
        plot.plot(epochs, second_tracks["loss"])
        plot.axvline(x=est2, color='r')
        plot.xlabel("epoch")
        plot.ylabel("loss (second phase)")
        plot.legend(["Empirical", "Theoretical time of 99% learning"])
        plot.savefig("results/singular_value_%.2f_adjusting.png" % s)
        
