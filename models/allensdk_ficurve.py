from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.glif_api import GlifApi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fontsize = 18

ctc = CellTypesCache()

glif_api = GlifApi()
list = glif_api.list_neuronal_models()

glif3_cell_ids = []
for l in list:
    if l["name"][0] == "3":
        glif3_cell_ids.append(l["id"])
print(len(glif3_cell_ids))

threshes = []
c_ms = []
asc_amps = []
asc_taus = []
asc_rs = []

for id in glif3_cell_ids:
    neuron_config = glif_api.get_neuron_configs([id])[id]

    threshes.append(neuron_config["init_threshold"])
    asc_amps.extend(neuron_config["asc_amp_array"])
    asc_taus.extend(neuron_config["asc_tau_array"])
    asc_rs.extend(neuron_config["coeffs"]["asc_amp_array"])
    c_ms.append(neuron_config["C"])
np.save("threshes_glif3_allensdk.npy", threshes)
np.save("cms_glif3_allensdk.npy", c_ms)
np.save("ascrs_glif3_allensdk.npy", asc_rs)
np.save("ascamps_glif3_allensdk.npy", asc_amps)
np.save("asctaus_glif3_allensdk.npy", asc_taus)

plt.hist(threshes, color = 'k', bins = 50)
plt.xlabel('threshold', fontsize = fontsize)
plt.ylabel('counts', fontsize = fontsize)
plt.savefig("threshold_glif3_allensdk")

plt.hist(c_ms, color = 'k', bins = 50)
plt.xlabel('C', fontsize = fontsize)
plt.ylabel('counts', fontsize = fontsize)
plt.savefig("capacitance_glif3_allensdk")

plt.hist(asc_amps, color = 'k', bins = 50)
plt.xlabel('asc_amp', fontsize = fontsize)
plt.ylabel('counts', fontsize = fontsize)
plt.savefig("asc_amp_glif3_allensdk")

plt.hist(asc_rs, color = 'k', bins = 50)
plt.xlabel('asc_r', fontsize = fontsize)
plt.ylabel('counts', fontsize = fontsize)
plt.savefig("asc_r_glif3_allensdk")

plt.hist(asc_taus, color = 'k', bins = 50)
plt.xlabel('asc_tau', fontsize = fontsize)
plt.ylabel('count', fontsize = fontsize)
plt.savefig("asc_tau_glif3_allensdk")

# fontsize = 18
# # download all electrophysiology features for all cells
# ephys_features = ctc.get_ephys_features()
# ef_df = pd.DataFrame(ephys_features)

# print("Ephys. features available for %d cells" % len(ef_df))

# plt.hist(ef_df['f_i_curve_slope'], color = 'k', bins = 50)
# plt.xlabel('f-i curve slope', fontsize = fontsize)
# plt.ylabel('counts', fontsize = fontsize)
# plt.show()