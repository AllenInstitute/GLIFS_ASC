from allensdk.core.cell_types_cache import CellTypesCache
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ctc = CellTypesCache()

fontsize = 18
# download all electrophysiology features for all cells
ephys_features = ctc.get_ephys_features()
ef_df = pd.DataFrame(ephys_features)

print("Ephys. features available for %d cells" % len(ef_df))

plt.hist(ef_df['f_i_curve_slope'], color = 'k', bins = 50)
plt.xlabel('f-i curve slope', fontsize = fontsize)
plt.ylabel('counts', fontsize = fontsize)
plt.show()