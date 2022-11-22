import json
import os

from visual_pomdp_smm.testing.test_utils import (
    test_function, calculate_std_table)

resultsDict = test_function(
    prefix_name_inputs=[
        'minigrid_uniform_latentspace_memory_8',
        'minigrid_uniform_latentspace_memory_32',
        'minigrid_uniform_latentspace_memory_256',
        ],
    save_figures=True,
    include_all_experiments=True
)

json_dict = json.dumps(resultsDict, indent=2, default=str)
print(json_dict)

if not os.path.exists("save"):
    os.makedirs("save")
# Writing to sample.json
with open(
        "save/Experiment_Test_Latent_Uniform_Memory.json",
        "w") as outfile:
    outfile.write(json_dict)

calculate_std_table(
    filename='save/Experiment_Test_Latent_Uniform_Memory.json')
