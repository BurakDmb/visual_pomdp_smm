import json
import os

import numpy as np
import natsort

with open("save/Experiment_Compare_Latent_Memory.json", "r") as infile:
    json_dict = json.load(infile)

    keys = list(json_dict.keys())
    unique_keys = set([key.rsplit('_', 1)[0] for key in keys])
    unique_keys = natsort.natsorted(unique_keys, reverse=False)
    results_dict = {}

    for unique_key in unique_keys:
        key_dict = {
            k: v for k, v in json_dict.items() if k.startswith(unique_key)}

        results_dict[unique_key] = {}
        results_dict[unique_key]['test_avgloss_mean'] = np.mean([
            v['test_avgloss'] for k, v in key_dict.items()])
        results_dict[unique_key]['test_avgloss_std'] = np.std([
            v['test_avgloss'] for k, v in key_dict.items()])

        results_dict[unique_key]['eval_avgloss_mean'] = np.mean([
            v['eval_avgloss'] for k, v in key_dict.items()])
        results_dict[unique_key]['eval_avgloss_std'] = np.std([
            v['eval_avgloss'] for k, v in key_dict.items()])

        results_dict[unique_key]['lossdiff_mean'] = np.mean([
            v['lossdiff'] for k, v in key_dict.items()])
        results_dict[unique_key]['lossdiff_std'] = np.std([
            v['lossdiff'] for k, v in key_dict.items()])

    print()
    json_result_dict = json.dumps(results_dict, indent=2, default=str)
    print(json_result_dict)

    if not os.path.exists("save"):
        os.makedirs("save")
    # Writing to sample.json
    with open(
            "save/Experiment_Compare_Latent_Memory_Stochastic_Results.json",
            "w") as outfile:
        outfile.write(json_result_dict)
