import numpy as np
import os
from fairseq import checkpoint_utils, tasks, utils, options
from fairseq.data import encoders
from analysis.align_accuracy.utils import get_hidden_states

np.set_printoptions(precision=10)


parser = options.get_generation_parser()
args = options.parse_args_and_arch(parser)
print(args)
task = tasks.setup_task(args)
task.load_dataset(args.gen_subset)


source = args.source_lang.split("_")[0]
print("====={}=====".format(args.key))
print("=====Start Loading=====")
state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
saved_args = state["args"]
model = task.build_model(saved_args)
model.load_state_dict(state["model"], strict=True)
model.cuda()
print("=====Start Calculating=====")
src_sents, src_values, indexes = get_hidden_states(task, model, args)
print("=====End=====")
os.system("mkdir -p {}/{}".format(args.savedir, args.key))
with open("{}/{}/{}.txt".format(args.savedir, args.key, source), "w") as fwe:
    for _id, src_w in enumerate(src_sents):
        fwe.write("{}\t{}\n".format(indexes[_id], src_w))

np.savetxt("{}/{}/{}_sent_avg_pool.csv".format(args.savedir, args.key, source), src_values,
           delimiter=',')

