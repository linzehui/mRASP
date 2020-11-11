import numpy as np
from tqdm import tqdm
from fairseq import checkpoint_utils, tasks, utils, options
from fairseq.data import encoders


def get_hidden_states(task, model, args):
    src_dict = getattr(task, 'source_dictionary', None)
    tgt_dict = task.target_dictionary
    
    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)
    
    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x
    
    def toks_2_sent(toks):
        _str = tgt_dict.string(toks, args.remove_bpe)
        _sent = decode_fn(_str)
        return _sent
    
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions()
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        # data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    
    # initialize
    src_sentences = []
    src_hidden_states_list = []
    idx_list = []
    
    for sample in tqdm(itr):
        sample = utils.move_to_cuda(sample)
        src_avg_states = get_avg(sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"], model, False)
        src_hidden_states_list.extend(src_avg_states)
        idx_list.extend(sample["id"].detach().cpu().numpy())
        for i, sample_id in enumerate(sample['id'].tolist()):
            src_tokens_i = utils.strip_pad(sample['net_input']['src_tokens'][i, :], src_dict.pad())
            src_sent_i = toks_2_sent(src_tokens_i)
            src_sentences.append(src_sent_i)
    return src_sentences, src_hidden_states_list, idx_list


def get_avg(tokens, lengths, model, has_langtok=False):
    encoder_outs = model.encoder.forward(tokens, lengths)
    np_encoder_outs = encoder_outs.encoder_out.detach().cpu().numpy().astype(np.float32)
    encoder_mask = 1 - encoder_outs.encoder_padding_mask.detach().cpu().numpy().astype(np.float32)
    encoder_mask = np.expand_dims(encoder_mask.T, axis=2)
    if has_langtok:
        encoder_mask = encoder_mask[1:, :, :]
        np_encoder_outs = np_encoder_outs[1, :, :]
    masked_encoder_outs = encoder_mask * np_encoder_outs
    avg_pool = (masked_encoder_outs / encoder_mask.sum(axis=0)).sum(axis=0)
    return avg_pool
