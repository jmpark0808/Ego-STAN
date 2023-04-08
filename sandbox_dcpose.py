from pose_models.DCPose_main.posetimation.config.defaults import _C
cfg = _C.clone()
cfg.merge_from_file('pose_models/DCPose_main/configs/posetimation/DcPose/posetrack18/model_RSN.yaml')
from pose_models.DCPose_main.posetimation.zoo import build_model
model = build_model(cfg, phase='train').cuda()
import torch
test = torch.zeros([1, 9, 288, 384]).cuda()
margin = torch.stack([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1)
print(margin.size())
margin = torch.cat([margin], dim=0).cuda()
print(margin.size())
result = model(test, margin=margin)
print(result.size())