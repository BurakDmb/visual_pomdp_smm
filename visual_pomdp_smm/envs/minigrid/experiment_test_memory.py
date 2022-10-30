import json
import os

from visual_pomdp_smm.testing.minigrid_test import test_function

resultsDict = test_function(
    test_dataset_class_str="MinigridMemoryUniformDatasetNoteval",
    eval_dataset_class_str="MinigridMemoryUniformDatasetEval",
    prefix_name_inputs=[
        'minigrid_uniform_memory_ae',
        'minigrid_uniform_memory_vae',
        'minigrid_uniform_memory_conv_ae',
        'minigrid_uniform_memory_conv_vae',
        'minigrid_uniform_memory_conv_binary_ae'],
    save_figures=True
)

json_dict = json.dumps(resultsDict, indent=2, default=str)
print(json_dict)

if not os.path.exists("save"):
    os.makedirs("save")
# Writing to sample.json
with open("save/Experiment_Test_Memory.json", "w") as outfile:
    outfile.write(json_dict)
