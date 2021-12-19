import torch

def mse(pred, label):
    pred = pred.reshape(pred.size(0), -1)
    label = label.reshape(label.size(0), -1)
    return torch.sum(torch.mean(torch.pow(pred-label, 2), dim=1))

def auto_encoder_loss(pose_pred, pose_label, hm_decoder, hm_resnet):
    lambda_p = 0.1
    lambda_theta = -0.01
    lambda_L = 0.5
    lambda_hm = 0.001
    pose_l2norm = torch.sqrt(torch.sum(torch.sum(torch.pow(pose_pred-pose_label, 2), dim=2), dim=1))
    cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
    cosine_similarity_error = torch.sum(cos(pose_pred, pose_label), dim=1)
    limb_length_error = torch.sum(torch.sum(torch.abs(pose_pred-pose_label), dim=2), dim=1)
    heatmap_error = torch.sqrt(torch.sum(torch.pow(hm_resnet.view(hm_resnet.size(0), -1) - hm_decoder.view(hm_decoder.size(0), -1), 2), dim=1))
    LAE_pose = lambda_p*(pose_l2norm + lambda_theta*cosine_similarity_error + lambda_L*limb_length_error)
    LAE_hm = lambda_hm*heatmap_error
    return torch.mean(LAE_pose), torch.mean(LAE_hm)
