import sys

if __name__ == "__main__":
    ckpts = sys.stdin
    keep = int(sys.argv[1]) if int(sys.argv[1]) > 0 else -1
    ckpt_list = [c for c in ckpts]
    if len(ckpt_list) <= keep:
        exit(0)
    elif len(ckpt_list) - keep == 1:
        print(ckpt_list[-1])
    else:
        exit(1)

