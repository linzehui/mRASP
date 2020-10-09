#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import torch
import os
import re


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for f in inputs:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state['model']

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        averaged_params[k].div_(num_models)
    new_state['model'] = averaged_params
    return new_state


def last_n_checkpoints(paths, n, update_based, upper_bound=None):
    assert len(paths) == 1
    path = paths[0]
    if update_based:
        pt_regexp = re.compile(r'checkpoint_\d+_(\d+)\.pt')
    else:
        pt_regexp = re.compile(r'checkpoint(\d+)\.pt')
    files = os.listdir(path)

    entries = []
    for f in files:
        m = pt_regexp.fullmatch(f)
        if m is not None:
            sort_key = int(m.group(1))
            if upper_bound is None or sort_key <= upper_bound:
                entries.append((sort_key, m.group(0)))
    if len(entries) < n:
        raise Exception('Found {} checkpoint files but need at least {}', len(entries), n)
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)[:n]]


def main():
    parser = argparse.ArgumentParser(
        description='Tool to average the params of input checkpoints to '
                    'produce a new checkpoint',
    )
    # fmt: off
    parser.add_argument('--inputs', required=True, nargs='+',
                        help='Input checkpoint files.')
    parser.add_argument('--output', required=True, metavar='FILE',
                        help='Write the new checkpoint containing the averaged weights to this path.')
    # fmt: on
    args = parser.parse_args()
    print(args)

    arg_inputs = args.inputs
    args_output = args.output

    average_checkpoints_fn(arg_inputs, args_output)


def average_checkpoints_fn(arg_inputs, args_output):
    print("checkpoints:{}\n"
          "output_path:{}\n".format(
        "\n\t"+"\n\t".join(arg_inputs),
        args_output,
    ), flush=True)
    new_state = average_checkpoints(arg_inputs)
    torch.save(new_state, args_output)
    print('Finished writing averaged checkpoint to {}.'.format(args_output), flush=True)


if __name__ == '__main__':
    main()
