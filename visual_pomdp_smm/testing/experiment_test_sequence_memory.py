import json
import os

import numpy as np

from visual_pomdp_smm.testing.test_utils import (
    test_function, calculate_std_table)

resultsDict, freq_vs_losses_dict = test_function(
    prefix_name_inputs=[
        'minigrid_sequence_memory_ae',
        'minigrid_sequence_memory_vae',
        # 'minigrid_sequence_memory_conv_ae',
        # 'minigrid_sequence_memory_conv_vae',
        # 'minigrid_sequence_memory_conv_binary_ae'
        ],
    save_figures=True,
    include_all_experiments=True
)

json_dict = json.dumps(resultsDict, indent=2, default=str)
print(json_dict)

if not os.path.exists("save"):
    os.makedirs("save")
# Writing to sample.json
with open("save/Experiment_Test_Sequence_Memory.json", "w") as outfile:
    outfile.write(json_dict)

np.save(
    "save/Experiment_Test_Sequence_Memory_Freq_Vs_Losses_Dict.npy",
    freq_vs_losses_dict)

calculate_std_table(filename='save/Experiment_Test_Sequence_Memory.json')
