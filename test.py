import argparse
import os
import time
import torch
import network as nw


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    cuda = device.type == "cuda"
    if cuda: nw.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))

    d_test = nw.datasets(args.dataset, args.data_dir, "val2017", train=True)

    print(args)
    num_classes = max(d_test.classes) + 1
    model = nw.snetwork(False, num_classes).to(device)

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    if cuda: torch.cuda.empty_cache()
    print("\nevaluating...\n")

    B = time.time()
    eval_output, iter_eval = nw.evaluate(model, d_test, device, args)
    B = time.time() - B

    print(eval_output.get_AP())
    if iter_eval is not None:
        print("\nTotal time of this evaluation: {:.1f} s, speed: {:.1f} imgs/s".format(B, 1 / iter_eval))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--data-dir", default="data/coco2017")
    parser.add_argument("--ckpt-path", default="network_coco-10.pth")
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    args.use_cuda = True
    args.results = os.path.join(os.path.dirname(args.ckpt_path), "network_results.pth")

    main(args)

