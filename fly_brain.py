# Author of this 2026 port to snnTorch: Michał Antropik Neuromorphicism
# Based on Brian2 model: https://github.com/philshiu/Drosophila_brain_model
# The network model for FlyWire is replicated from:  
# P. K. Shiu, G. R. Sterne, N. Spiller et al. “A drosophila computational brain model reveals sensorimotor processing” Nature, vol. 634, no. 8032, pp. 210–219, 2024.
# My work was partly inspired by this paper that moved the Brian2 model to STACS Charm++ model but without a public code repository: https://arxiv.org/html/2508.16792v1

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import snntorch as snn

# You can also import:
# auditory_sound_sensing_ammc_neurons
# visual_optic_lobe_sensing_mi1_neurons
# taste_sugar_sensing_neurons
# olfaction_smell_sensing_neurons

from sensory_neurons import taste_sugar_sensing_neurons

config = {
    'path_completeness': './data/2023_03_23_completeness_630_final.csv',             # csv of the complete list of Flywire neurons 127K
    'path_connectivity': './data/2023_03_23_connectivity_630_final.parquet',         # connectivity data
    'path_saved_weights': './model/fly_brain_connections_weights.pth',               # build this file here below if you don't have it
    'n_proc': -1,                                                                    # number of CPU cores (-1: use all)
}

# Options to SAVE / LOAD model weights from your own hard drive set both to True if you need it in a file
save_model_weights_to_hard_drive = False
load_model_weights_from_hard_drive = False

device = torch.device("cpu")

# Uncomment the block below if you have enough RAM or VRAM to run this model on Apple or NVIDIA hardware
# 280 MB for sparse version
# 60 GB for dense version

"""
if torch.backends.mps.is_available():
    # Won't probably work on Apple mps unless code is modified to use sparse CSR format instead of COO and there is no addmv_ (@)
    # Apple CPU MLX Sparse Solvers work much better with CSR at the moment
    device = torch.device("mps")
"""

"""
if torch.cuda.is_available():
    # Same CSR requirements as in MPS
    device = torch.device("cuda")
"""

print("\nThe model will run on device: ", device)


# snnTorch LIF does not support the conductance variable, continuous-time ODEs, Brian2-style refractory counters, PoissonInput objects

params = {
    "dt": 1.0,                    # ms best 1.0
    "tau_membrane": 20.0,         # 20 ms per https://arxiv.org/html/2508.16792v1 this is only used to calculate beta below TRY 5 ms for faster leak
    "threshold_level": 1.0,       # threshold of neuron was -45 mV TRY 1.0 or 0.83
    "reset_level": 0.0,           # reset was -52 mV this can only be 0.0 in snnTorch LIF neuron with use of reset_mechanism
    "weight_per_synapse": 0.05,   # constant base weight of all synapses was 0.275 TRY 0.05 or 0.15

    "rate_poisson": 150.0,        # 150 Hz per https://arxiv.org/html/2508.16792v1
    "frequency_poisson": 250.0,   # 250 Hz original value in Brian2 implementation
    
    "timesteps_to_run": 1000,     # adjust it to 100 on weak CPU or to 100000 if you have a good GPU
}


# snnTorch LIF neuron is used here but a model with snnTorch Lapicque or Synaptic or Alpha can also be implemented

LIF_NEURON = snn.Leaky(
    beta = 1 - (params["dt"] / params["tau_membrane"]),
    threshold = params["threshold_level"],
    reset_mechanism = "zero",
).to(device)


# Connectivity of all fruit fly brain neurons as vectorized function
# If Python for loop would be used it would take around 4 minutes to build those dense connections on M2 Max instead of 10 seconds

def build_weights_matrix(neurons_connections, number_of_neurons, weight_per_synapse):
    
    begin_time = datetime.datetime.now()
    print("\nBuilding connections... started at: ", begin_time)

    # Extract columns as tensors
    presynaptic_index  = torch.tensor(neurons_connections["Presynaptic_Index"].values, dtype=torch.long, device=device)
    postsynaptic_index = torch.tensor(neurons_connections["Postsynaptic_Index"].values, dtype=torch.long, device=device)

    # Some weights might be negative, which represent the inhibitory synapses
    # TODO: This probably could be optimized through the calculated single weight from different weights in same pre/post connections
    # If one neuron receives multiple inputs then those can be joined with a method described in: https://arxiv.org/html/2508.16792v1
    final_weights = torch.tensor(neurons_connections["Excitatory x Connectivity"].values, device=device) * weight_per_synapse



    # WORSE DENSE OPTION: Allocate  matrix
    #connections_weights = torch.zeros((number_of_neurons, number_of_neurons), device=device)

    # WORSE DENSE OPTION: Vectorized assignment
    #connections_weights[postsynaptic_index, presynaptic_index] = final_weights

    # I am leaving this worse option available to show people that sparse neuromorphic SNNs are the future of ML in terms of speed, size and general results

    
    # OPTIMIZED FASTER SPARSE OPTION: Build COO indices
    indices = torch.stack([postsynaptic_index, presynaptic_index])

    # OPTIMIZED FASTER SPARSE OPTION: Build sparse matrix
    connections_weights = torch.sparse_coo_tensor(
        indices,
        final_weights,
        size=(number_of_neurons, number_of_neurons),
        device=device
    ).coalesce()

    end_time = datetime.datetime.now()
    print("\nDone building connections... elapsed time: ", end_time - begin_time, "\n")


    # WARNING! This file if saved will weight 280 MB in sparse COO format or 60 GB in dense format on your computer!
    if save_model_weights_to_hard_drive:
        torch.save(connections_weights, config['path_saved_weights'])
        print(f"\nSaved weights to {config['path_saved_weights']}\n")

    return connections_weights


# Poisson input generator is implemented here instead of Brian2 PoissonInput
# snnTorch spikegen could also be used here

def poisson_input(number_of_neurons, rate_hz, weight, dt_ms):
    p = rate_hz * (dt_ms / 1000.0)
    spikes = (torch.rand(number_of_neurons, device=device) < p).float()
    return spikes * weight


# Full fruit fly brain simulation in snnTorch with Poisson inputs

def run_spiking_neural_network(neurons_list, neurons_connections, params):
    number_of_neurons = len(neurons_list)
    steps = int(params["timesteps_to_run"] / params["dt"])

    # .to_sparse_csr() speeds up the run significantly because COO is optimized for building sparse matrices not running them
    # This port now surely beats the Brian2 implementation in terms of simulation speed and might reach the Loihi 2 implementation speed shown in: https://arxiv.org/html/2508.16792v1
    # WARNING! .to_sparse_csr() is still in beta in PyTorch so remove it if the code won't work at all

    weights_matrix = None

    if load_model_weights_from_hard_drive == False:
        # WARNING! Build weights matrix first before loading the saved fly_brain_connections_weights.pth
        weights_matrix = build_weights_matrix(neurons_connections, number_of_neurons, params["weight_per_synapse"]).to_sparse_csr()
    else:
        # WARNING! You can also load 60 GB / 280 MB of weights from the previously saved pth file
        weights_matrix = torch.load(config['path_saved_weights'], map_location="cpu").to_sparse_csr()
        print(f"\nLoaded weights from {config['path_saved_weights']}\n")


    # Change it to True or False if you want to see real biological simulation
    run_sensory_experiment = True
    sensory_neurons_indexes = None
    mask = None

    # BEGIN SENSORY EXPERIMENT SETUP

    if run_sensory_experiment:
        
        neurons_list = pd.read_csv(config['path_completeness'], index_col=0)

        # You can change taste_sugar_sensing_neurons to auditory_sound_sensing_ammc_neurons or other stated under line 15 and run different experiments
        # Default is taste sugar experiment
        chosen_sensory_neurons = taste_sugar_sensing_neurons

        if len(chosen_sensory_neurons) == 0:
            print("\nSensory neurons list is empty!\n")

        # Find neurons indexes in csv neurons_list
        sensory_neurons_indexes = [
            neurons_list.index.get_loc(neuron_id)
            for neuron_id in chosen_sensory_neurons
            if neuron_id in neurons_list.index
        ]

        print("\nFound chosen sensory neurons indexes: ", sensory_neurons_indexes, "\n")
            

        # Mask for stimulated sensory neurons
        stimulated_sensory_neurons = torch.tensor(sensory_neurons_indexes, device=device)

        # This could also be turned to the sparse version
        mask = torch.zeros(number_of_neurons, device=device)
        mask[stimulated_sensory_neurons] = 1.0

    # END SENSORY EXPERIMENT SETUP



    # Network state variables
    membrane_potential = torch.full((number_of_neurons,), params["reset_level"], device=device)
    previous_spikes = torch.zeros(number_of_neurons, device=device)
    spikes_record = torch.zeros((steps, number_of_neurons), device=device)
    


    # Generate random spikes and input those into fruit fly brain SNN
    # dt = 0.1 and timesteps_to_run = 100 so 1000 dense steps took 4 hours on M2 Max CPU mode
    # 1000 sparse COO steps took 5 minutes on M2 Max CPU mode
    # 1000 sparse CSR steps took 5 seconds on M2 Max CPU mode
    for step in range(steps):

        # Synaptic input
        I_synaptic = weights_matrix @ previous_spikes

        # Poisson input
        I_poisson_full = poisson_input(
            number_of_neurons,
            params["rate_poisson"],
            params["weight_per_synapse"] * params["frequency_poisson"],
            params["dt"]
        )

        # SENSORY EXPERIMENT: Apply mask so only chosen neurons get the Poisson input
        if run_sensory_experiment:
            I_poisson_full = I_poisson_full * mask

        # Without applied mask all brain neurons will be stimulated
        total_spiking_input = I_synaptic + I_poisson_full

        print(f"Input spikes #{step}: ", total_spiking_input)

        # LIF update
        resulting_spikes, membrane_potential = LIF_NEURON(total_spiking_input, membrane_potential)
        previous_spikes = resulting_spikes
        
        spikes_record[step] = resulting_spikes


    return spikes_record.squeeze().detach().cpu().numpy()




# Load completeness: 127 500 neurons

neurons_list = pd.read_csv(config['path_completeness'], index_col=0)

print("\nLoaded the list of all available fruit fly brain neurons!")

# Load connectivity rows: 14 687 178 synapses

neurons_connections = pd.read_parquet(config['path_connectivity'])

print("\nLoaded connections data for all neurons!\n")

spikes = run_spiking_neural_network(neurons_list, neurons_connections, params)

print("\nFinished running fruit fly brain snn simulation: ", spikes)



# Population firing rate / activity

rate = spikes.mean(axis=1)

plt.plot(rate)
plt.xlabel(f"Timestep ({params['dt']}ms)")
plt.ylabel("Mean firing rate")
plt.title("Population firing rate")
plt.show()


# Raster plot

plt.figure(figsize=(10, 8))
spike_times, neuron_ids = np.where(spikes == 1)

plt.scatter(spike_times, neuron_ids, s=0.01, color="black")
plt.xlabel(f"Timestep ({params['dt']}ms)")
plt.ylabel("Neuron index")
plt.title("Raster plot of spikes")
plt.show()


# Spike count histogram

plt.figure(figsize=(10, 6))
spike_counts = spikes.sum(axis=0)
plt.hist(spike_counts, bins=5)
plt.title("Spike count distribution")
plt.xlabel(f"Timestep ({params['dt']}ms)")
plt.ylabel("Spikes")
plt.show()
