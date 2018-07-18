import numpy as np
import matplotlib.pyplot as plot
from orthogonal_matrices import random_orthogonal

############ config
num_runs = 1
num_input = 100
num_output = 400 
rank = 2 # other ranks not supported yet 
num_epochs = 10000
esses = [10, 5, 3]
s_new = 2
init_size = 1e-5
epsilon = 0.01
overlap = 0.9
lr = 1e-3
second_start_time = 3000
#############
tau = 1./lr 

def _train(sigma_31, sigma_11, W21, W32, num_epochs, track_mode_alignment=False,
           new_input_modes=None, new_output_modes=None,):
    tracks = {
#        "W21": np.zeros([num_epochs, num_input]),
#        "W32": np.zeros([num_epochs, num_output]),
        "loss": np.zeros([num_epochs+1])
        }
    if track_mode_alignment:
        tracks["alignment"] = np.zeros([num_epochs+1])

    l = sigma_31 - np.dot(W32, np.dot(W21, sigma_11))
    tracks["loss"][0] = np.sum(np.square(l))
    for epoch in range(1, num_epochs + 1):
        l = sigma_31 - np.dot(W32, np.dot(W21, sigma_11))
        W21 += lr * np.dot(W32.transpose(), l) 
        W32 += lr * np.dot(l, W21.transpose()) 
#        tracks["W21"][epoch, :] = W21
#        tracks["W32"][epoch, :] = W32
        tracks["loss"][epoch] = np.sum(np.square(l))
        if track_mode_alignment:
            vec0, vec1 = _get_rep_modes(W21, W32, new_input_modes,
                                        new_output_modes, orthogonal=False)
            tracks["alignment"][epoch] = np.dot(vec0, vec1)

    return W21, W32, tracks

def _ugly_function(sc, c0, s, theta):
    return np.log((sc + c0 + s*np.tanh(theta/2.))/(sc - (c0 + s*np.tanh(theta/2.))))

def _estimated_learning_time(a0, b0, s, epsilon, tau):
    c0 = 2*np.abs(a0**2 - b0**2) 
    theta0 = np.arcsinh(a0*b0/c0)
    thetaf = np.arcsinh((1-epsilon) * s/ c0)

    sc = np.sqrt((c0)**2 + (s)**2)

    t = (0.5*tau/sc)*(_ugly_function(sc, c0, s, thetaf) - _ugly_function(sc, c0, s, theta0))
    return t

def _estimated_learning_times(a0, b0, s, tau, num_points=1000):
    start = a0*b0/s
    end = 1.0
    alignments = np.arange(start, end, (1./num_points) *(end-start) ) 
    epsilons = 1.-alignments
    times = np.zeros(len(alignments))
    for i in range(1, len(alignments)):
        times[i] = _estimated_learning_time(a0, b0, s, epsilons[i], tau)
    
    return times, epsilons

def _get_rep_modes(W21, W32, new_input_modes, new_output_modes, orthogonal=True):
    a0s = np.dot(W21, new_input_modes.transpose())
    b0s = np.dot(W32.transpose(), new_output_modes.transpose()) 
    vec0 = a0s[:, 0] * b0s[:, 0]
    vec0 *= np.sign(a0s[:, 0])
    vec0 /= np.linalg.norm(vec0)
    if orthogonal:
        vec1 = np.copy(vec0)[::-1] 
        vec1[1] *= -1
    else:
        vec1 = a0s[:, 1] * b0s[:, 1]
        vec1 *= np.sign(a0s[:, 1])
        vec1 /= np.linalg.norm(vec1)
    return vec0, vec1

def _coefficients_from_weights_and_modes(W21, W32, new_input_modes, new_output_modes):
    a0s = np.dot(W21, new_input_modes.transpose())
    b0s = np.dot(W32.transpose(), new_output_modes.transpose()) 
    vec0, vec1 = _get_rep_modes(W21, W32, new_input_modes, new_output_modes, True)
    a00 = np.sum((a0s[:, 0]**2)*vec0)
    a00 = np.sign(a00) * np.sqrt(np.abs(a00))
    b00 = np.sum((b0s[:, 0]**2)*vec0)
    b00 = np.sign(b00) * np.sqrt(np.abs(b00))
    index2 = 2 if new_mode == "orthogonal" else 1
    a01 = np.sum((a0s[:, index2]**2)*vec1)
    a01 = np.sign(a01) * np.sqrt(np.abs(a01))
    b01 = np.sum((b0s[:, index2]**2)*vec1)
    b01 = np.sign(b01) * np.sqrt(np.abs(b01))
    return a00, b00, a01, b01

for run_i in range(num_runs):
    for new_mode in ["partially_aligned", "orthogonal"]:
        for s in esses:
            new_input_modes = random_orthogonal(num_input)[0:3, :]
            new_output_modes = random_orthogonal(num_output)[0:3, :]
            original_input_mode = overlap * new_input_modes[0:1, :] + np.sqrt(1-overlap**2) *  new_input_modes[1:2, :]
            original_output_mode = overlap * new_output_modes[0:1, :] + np.sqrt(1-overlap**2) *  new_output_modes[1:2, :]
            if new_mode == "orthogonal":
                 S_new =  np.diag([s, 0, s_new])
            else:
                 S_new =  np.diag([s, s_new, 0])

            input_data = np.eye(num_input) # random_orthogonal(num_input)

            sigma_31 = np.dot(original_output_mode.transpose(), s*np.dot(original_input_mode, input_data))
            new_sigma_31 = np.dot(new_output_modes.transpose(), np.dot(S_new, np.dot(new_input_modes, input_data)))
            sigma_11 = np.eye(num_input)
            
            # initial weights
            W21 = np.sqrt(init_size)*random_orthogonal(num_input)[:rank, :]
            W32 = np.sqrt(init_size)*random_orthogonal(num_output)[:, :rank]

            # learning from random init -- theory
            a0s = np.dot(W21, original_input_mode.transpose())
            b0s = np.dot(W32.transpose(), original_output_mode.transpose()) 
            a0 = np.sqrt(np.sum(a0s**2))
            b0 = np.sqrt(np.sum(b0s**2))
            est1_times, est1_epsilons = _estimated_learning_times(a0, b0, s, tau)
            est1_init_loss = s**2 # initial outputs ~= 0 

            # learning from random init -- empirical
            W21, W32, first_tracks = _train(sigma_31, sigma_11, W21, W32, num_epochs)   
#        print(est1)
#        est1_int = int(est1)
#        print(np.dot(first_tracks["W21"][est1_int, :], original_mode))
#        print(first_tracks["W32"][est1_int, :])
#        print(s * (1-epsilon))
            
            # updating to new situation -- theory
            a00, b00, a01, b01 = _coefficients_from_weights_and_modes(W21, W32, new_input_modes, new_output_modes)
            est2_0_times, est2_0_epsilons = _estimated_learning_times(a00, b00, s, tau)
            est2_1_times, est2_1_epsilons = _estimated_learning_times(a01, b01, s_new, tau)
            
            est2_0_init_loss = s**2 *2 *(1 - overlap**2) if new_mode == "orthogonal" else 2 *( s**2 - s * s_new) *  (1 - overlap**2)  
            est2_1_init_loss = (s_new)**2

            # updating to new situation --empirical  

            W21, W32, second_tracks = _train(new_sigma_31, sigma_11, W21, W32, second_start_time,
                                             True, new_input_modes, new_output_modes)

            # updating to new situation -- semi-theory starting from after first mode learning is approximately done
            # still isn't perfect because modes aren't truly orthogonal until too late in the learning process
            _, _, a01, b01 = _coefficients_from_weights_and_modes(W21, W32, new_input_modes, new_output_modes)
            est2_1_times, est2_1_epsilons = _estimated_learning_times(a01, b01, s_new, tau)
            est2_1_times += second_start_time # offset

            W21, W32, second_tracks_2 = _train(new_sigma_31, sigma_11, W21, W32, num_epochs-second_start_time,
                                             True, new_input_modes, new_output_modes)

            for key in second_tracks.keys():
                second_tracks[key] = np.concatenate([second_tracks[key], second_tracks_2[key][1:]], 0)

            # plotting
            adjusting_loss = est2_0_epsilons/np.amax(est2_0_epsilons)*est2_0_init_loss
            new_loss = est2_1_epsilons/np.amax(est2_1_epsilons)*est2_1_init_loss
            epochs = range(num_epochs + 1)
            approx_summed_loss = np.zeros_like(epochs, np.float32)  
            for i, epoch in enumerate(epochs):
                this_index = np.argmin(np.abs(est2_0_times- epoch)) 
                approx_summed_loss[i] = adjusting_loss[this_index] 
                this_index_2 = np.argmin(np.abs(est2_1_times- epoch)) 
                approx_summed_loss[i] += new_loss[this_index_2] 

            plot.figure()
            plot.plot(epochs, first_tracks["loss"])
            plot.plot(est1_times, est1_epsilons/np.amax(est1_epsilons)*est1_init_loss, color='r')
            plot.xlabel("Epoch")
            plot.ylabel("Loss (first phase)")
            plot.legend(["Empirical", "Theory"])
            plot.savefig("results/singular_value_%.2f_condition_%s_initial_learning.png" % (s, new_mode))
            plot.figure()
            plot.plot(epochs, second_tracks["loss"])
            plot.plot(est2_0_times, adjusting_loss)
            plot.plot(est2_1_times, new_loss)
            plot.plot(epochs, approx_summed_loss)
            plot.xlabel("Epoch")
            plot.ylabel("Loss (second phase)")
            plot.legend(["Empirical", "Theory (adjusted mode)", "Theory (new mode)", "Theory (total)"])
            plot.savefig("results/singular_value_%.2f_condition_%s_adjusting.png" % (s, new_mode))
            plot.figure()
            plot.plot(epochs, second_tracks["alignment"], color='#550055')
            plot.xlabel("Epoch")
            plot.ylabel("Empirical representation alignment")
            plot.savefig("results/singular_value_%.2f_condition_%s_adjusting_rep_alignment.png" % (s, new_mode))
