from typing import NamedTuple, Optional

"""
Defines parameters regarding training process
"""
class TrainParameters(NamedTuple):
    network_type: str
    batch_size: int
    lr: float
    num_epochs: int
    log_dir: str

"""
Defines properties of network
"""
class NetworkParameters(NamedTuple):
    dropout_prob: float = 0 # Dropout probability
    output_weight: bool = False # Whether the output of neuron should be weighted
    delay: float = 0 # Synaptic delay in msec

"""
Defines structure of a network
"""
class StructureParameters(NamedTuple):
        input_size: int
        hidden_size: int
        output_size: int

"""
Parameters regarding neuronal computation
"""
class NeuronParameters(NamedTuple):
    # Simulation-related
    dt: float = 0.05
    tau: float = 0.05

    # Specific to GLIFR
    num_ascs: Optional[int] = 2
    sigma_v: Optional[float] = 1
    R: Optional[float] = 0.1
    I0: Optional[float] = 0
    v_reset: Optional[float] = 0
    initialization: Optional[str] = 'hom' # hom or het