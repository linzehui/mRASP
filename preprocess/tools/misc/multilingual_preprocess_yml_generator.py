# get auxillary file path and modify the yml file
import yaml
import sys
import os

"""
This script generate temporary preprocess yaml file for a language pair
args:
    configs_path: output yaml file to this path
    main_yml_file: main yaml
    key: string that indicates lang / lang pair (with '_')
    prefix: the prefix of the yaml file, default: "preprocess", final yaml name: `prefix`_`key`
    file_prefix: the prefix for filename; for example, "train" "dev" or "test"
"""
main_yml_file = sys.argv[1]
key = sys.argv[2]
file_prefix = sys.argv[3]
prefix = "preprocess"


def get_paths(params):
    _logs_path = os.path.join(params["output_main_path"], params["logs_subdir"])
    _configs_path = os.path.join(params["output_main_path"], params["configs_subdir"])
    _cleaned_path = os.path.join(params["output_main_path"], params["configs_subdir"])
    
    return _configs_path, _logs_path, _cleaned_path


def generate_mono(config, key):
    if key not in config:
        _config = config["default_langs"]
    else:
        _config = config[key]
    _config["language"] = key
    return _config


def generate_pair(config, key):
    l = key.split('_')
    _l = key.split('2')
    subdir = key
    if file_prefix != "train":  # len(_l) == 2
        _subdir = "2".join([_l[1], _l[0]])
        langs = _l
    else:  # len(l) == 2
        _subdir = "_".join([l[1], l[0]])
        langs = l
    key = "_".join([langs[0], langs[1]])
    _key = "_".join([langs[1], langs[0]])
    data_path = config["raw_data_path"]

    configs_path, logs_path, cleaned_path = get_paths(config)
    
    # get the right subdir that contains parallel data
    if not os.path.isfile(os.path.join(data_path, subdir, file_prefix + "." + langs[0])):
        if not os.path.isfile(os.path.join(data_path, _subdir, file_prefix + "." + langs[0])):
            raise Exception("No files {} in {} or {}!".format(file_prefix + "." + langs[0], subdir, _subdir))
        else:
            subdir = _subdir
            
    if key not in config:
        # switch the order of src and trg
        key = _key
    if key not in config:
        key = "default_pairs"
        
    _config = config[key]  # shuffle, deduplicate, keep_lines_percent
    _config["log_file"] = os.path.join(logs_path, "{}_{}.log".format(prefix, subdir))
    for s, lang in zip(["language1", "language2"], langs):
        _config[s] = {}
        _config[s]["language"] = lang
        _config[s]["config_file"] = os.path.join(configs_path, prefix + '_' + lang + ".yml")
        _config[s]["file"] = os.path.join(data_path, subdir, file_prefix + "." + lang)
    _config["output_path"] = os.path.join(cleaned_path, subdir)
    return _config


def generate_config(config, key):
    l = key.split('_')
    _l = key.split('2')
    if len(l) == 1 and len(_l) == 1:  # lang
        return generate_mono(config, key)
    elif len(l) == 2 or len(_l) == 2:  # lang pair
        return generate_pair(config, key)
    else:
        raise Exception("Invalid key {}!".format(key))


if __name__=="__main__":
    with open(main_yml_file, 'r') as fyml:
        params = yaml.safe_load(fyml)  # get params from yaml
        config = generate_config(params, key)
    configs_path, _, _ = get_paths(params)
    config_file = os.path.join(configs_path, "preprocess_" + key + ".yml")
    os.system("mkdir -p {}".format(configs_path))
    with open(config_file, 'w') as fw:
        yaml.dump(config, fw, default_flow_style=False)

