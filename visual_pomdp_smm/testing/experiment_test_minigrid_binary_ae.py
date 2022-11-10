import json
import os

from visual_pomdp_smm.testing.test_utils import test_function

resultsDict = test_function(
    test_dataset_class_str='MinigridMemoryFullDataset',
    eval_dataset_class_str='MinigridMemoryKeyDataset',
    prefix_name_inputs=[
        'minigrid_memory_binary_AE_256',
        'minigrid_memory_binary_AE_128',
        'minigrid_memory_binary_AE_64',
        'minigrid_memory_binary_AE_32',
        'minigrid_memory_binary_AE_16'],
    save_figures=True
)

json_dict = json.dumps(resultsDict, indent=2, default=str)
print(json_dict)

if not os.path.exists("save"):
    os.makedirs("save")
# Writing to sample.json
with open("save/Experiment_Test_Memory.json", "w") as outfile:
    outfile.write(json_dict)
