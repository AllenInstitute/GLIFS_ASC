# GLIFS_ASC
GLIFS plus after spike currents

Goal: To create a rate-based neuronal model that incorporates after-spike currents, to connect these in networks, and to train the model on tasks, learning neuronal parameters (e.g., threshold, after-spike current related terms) in addition to synaptic weights.

Important files:
- models/neurons/glif_new: defines a GLIFR class for the rate based neuronal model, a RNNC class for a vanilla RNN cell, and a Placeholder and PlaceholderWt classes for placeholder passing of data
- models/neurons/network_new: defines a neural network using GLIFRs and Placeholders as a biologicall realistic neural network and a neural network using RNNCs and Placeholders
- models/neurons/utils_glif_new: helper functions for updating neuronal states ot be used by glif_new
- models/pattern_generation.py: program for testing sinusoid generation in three paradigms
- models/pattern_generation_rnn.py: program for testing sinusoid generation in a vanilla RNN
