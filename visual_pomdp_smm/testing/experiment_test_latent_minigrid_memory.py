import json
import os

from visual_pomdp_smm.testing.test_utils import (
    test_function, calculate_std_table)

resultsDict = test_function(
    prefix_name_inputs=[
        'minigrid_memory_conv_binary_AE_16',
        'minigrid_memory_conv_binary_AE_32',
        'minigrid_memory_conv_binary_AE_64',
        'minigrid_memory_conv_binary_AE_128',
        'minigrid_memory_conv_binary_AE_256',
        'minigrid_memory_conv_AE_16',
        'minigrid_memory_conv_AE_32',
        'minigrid_memory_conv_AE_64',
        'minigrid_memory_conv_AE_128',
        'minigrid_memory_conv_AE_256',
        ],
    save_figures=True,
    include_all_experiments=True
)

json_dict = json.dumps(resultsDict, indent=2, default=str)
print(json_dict)

if not os.path.exists("save"):
    os.makedirs("save")
# Writing to sample.json
with open("save/Experiment_Compare_Latent_Memory.json", "w") as outfile:
    outfile.write(json_dict)


calculate_std_table(filename='save/Experiment_Compare_Latent_Memory.json')
