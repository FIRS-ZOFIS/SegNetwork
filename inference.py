import torch
import network as nw
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

use_cuda = True
dataset = "coco"
ckpt_path = "pretrained.pth"  # 预训练文件名
data_dir = "data/coco2017/"

device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    nw.get_gpu_prop(show=True)
print("\ndevice: {}".format(device))

ds = nw.datasets(dataset, data_dir, "val2017", train=True)
d = torch.utils.data.DataLoader(ds, shuffle=False)

model = nw.snetwork(True, max(ds.classes) + 1).to(device)
model.eval()
model.head.score_thresh = 0.3

if ckpt_path:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(checkpoint["eval_info"])
    del checkpoint

for p in model.parameters():
    p.requires_grad_(False)

iters = 3

for i, (image, target) in enumerate(d):
    image = image.to(device)[0]

    with torch.no_grad():
        result = model(image)
    nw.show(image, result, ds.classes, "/image/output{}.jpg".format(i))
    if i >= iters - 1:
        break
