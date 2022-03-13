import json
import os

def config_experiments(results_dir, create_json=True):

    with open('./base_config.json') as config_file:
        base_config = json.load(config_file)

    id = 0
    experiment_list = []
    dataset=["cifar10", "cifar10c"]
    for net in ["ResNet", "AlexNet"]:
        for data in dataset:
                config = base_config.copy()
                config["data"] = data
                config["net"] = net

                if create_json:
                    with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                        json.dump(config, json_file)
                experiment_list.append(config.copy())
                id += 1



    print(str(id) + " config files created")
    return experiment_list