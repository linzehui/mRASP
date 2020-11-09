import argparse
import itertools
import torch
import typing
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--now-vocab", type=str, required=True)
    parser.add_argument("--to-append-vocab", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def _load_checkpoint(path: str):
    state = torch.load(
        path,
        map_location=(
            lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
        ),
    )
    return state


def _load_vocabs(path: str):
    res = {}
    with open(path, "r") as f:
        for l in f:
            token, number = l.strip().split(" ")
            res[token] = number
    return res


def _append_vocab_to_checkpoint(model_states: typing.Dict[str, typing.Any], to_append_vocab: typing.List[str]):
    args = model_states['args']
    model = model_states['model']
    assert getattr(args, "share_all_embeddings")


    def _move_tensor(x):
        embedding_dim = x.size(1)
        append_vocab = x.new_ones(size=(len(to_append_vocab), embedding_dim)).float()
        torch.nn.init.normal_(append_vocab, mean=0, std=embedding_dim ** -0.5)
        append_vocab = append_vocab.half()
        x = torch.cat((x, append_vocab, ), dim=0)
        
        return x

    model['encoder.embed_tokens.weight'] = _move_tensor(model['encoder.embed_tokens.weight'])
    # model['decoder.embed_out'] = _move_tensor(model['decoder.embed_out'])
    model['decoder.embed_tokens.weight'] = _move_tensor(model['decoder.embed_tokens.weight'])
    return model_states


def main():
    args = get_args()
    checkpoint = args.checkpoint
    now_vocab = args.now_vocab
    to_append_vocab = args.to_append_vocab
    output_dir = args.output_dir

    now_vocab = _load_vocabs(now_vocab)
    print(f"| Now vocab size is: {len(now_vocab)}")
    to_append_vocab = _load_vocabs(to_append_vocab)
    print(f"| New vocab size is: {len(to_append_vocab)}")
    to_append_vocab = dict(filter(lambda x: x[0] not in now_vocab, to_append_vocab.items()))
    to_append_vocab = dict(map(lambda x: (x[0], 20), to_append_vocab.items()))
    print(f"| After removing existed tokens, the new vocab size is: {len(to_append_vocab)}")

    states = _load_checkpoint(checkpoint)
    states = _append_vocab_to_checkpoint(states, list(to_append_vocab.keys()))
    os.system("mkdir -p {}".format(output_dir))
    
    # save checkpoint
    torch.save(states, os.path.join(output_dir, "model.pt"))
    
    # new vocab
    with open(os.path.join(output_dir, "dict.txt"), "w") as f:
        for k, v in itertools.chain(
            now_vocab.items(),
            to_append_vocab.items(),
        ):
            f.write(f"{k} {v}\n")


if __name__ == '__main__':
    main()
