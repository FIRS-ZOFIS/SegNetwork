import bisect
import glob
import os
import re
import time
import torch
import network as nw


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda": 
        nw.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
    dataset_train = nw.datasets(args.dataset, args.data_dir, "train2017", train=True)
    indices = torch.randperm(len(dataset_train)).tolist()
    d_train = torch.utils.data.Subset(dataset_train, indices)
    d_test = nw.datasets(args.dataset, args.data_dir, "val2017", train=True)
    args.warmup_iters = max(1000, len(d_train))
    print(args)
    num_classes = max(d_train.dataset.classes) + 1
    model = nw.snetwork(True, num_classes).to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_lambda = lambda x: 0.1 ** bisect.bisect(args.lr_steps, x)
    start_epoch = 0
    prefix, ext = os.path.splitext(args.ckpt_path)
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]
        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))

    # 训练
    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))
            
        A = time.time()
        args.lr_epoch = lr_lambda(epoch) * args.lr
        print("lr_epoch: {:.5f}, factor: {:.5f}".format(args.lr_epoch, lr_lambda(epoch)))
        iter_train = nw.train_one_epoch(model, optimizer, d_train, device, epoch, args)
        A = time.time() - A
        B = time.time()
        eval_output, iter_eval = nw.evaluate(model, d_test, device, args)
        B = time.time() - B
        trained_epoch = epoch + 1
        print("training: {:.1f} s, evaluation: {:.1f} s".format(A, B))
        nw.collect_gpu_info("network", [1 / iter_train, 1 / iter_eval])
        print(eval_output.get_AP())

        nw.save_ckpt(model, optimizer, trained_epoch, args.ckpt_path, eval_info=str(eval_output))
        prefix, ext = os.path.splitext(args.ckpt_path)
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        n = 10
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.system("rm {}".format(ckpts[i]))

    print("\ntotal time of this training: {:.1f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))
    
    
if __name__ == "__main__":
    import argparse
    # 参数列表
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--data-dir", default="/data/coco2017")  # 修改为coco数据路径
    parser.add_argument("--ckpt-path")
    parser.add_argument("--results")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[6, 7])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--print-freq", type=int, default=100)
    args = parser.parse_args()
    
    if args.lr is None:
        args.lr = 0.00125
    if args.ckpt_path is None:
        args.ckpt_path = "./network_{}.pth".format(args.dataset)
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "network_results.pth")
    
    main(args)
    
    