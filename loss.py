import torch
import cv2

def flow_loss_func(flow_preds, flow_gt, valid,
                   gamma=0.9,
                   max_flow=400,
                   **kwargs,
                   ):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # [B, H, W]
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        i_loss = (flow_preds[i] - flow_gt).abs()

        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()

    if valid.max() < 0.5:
        pass

    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    return flow_loss, metrics

def apply_mask(flow_preds):
    mask = cv2.imread('mask_672.png', cv2.IMREAD_GRAYSCALE)
    mask[mask>0]=1
    mask = torch.tensor(mask, device=flow_preds[0].device)
    mask = torch.stack((mask, mask), dim=-1).unsqueeze(dim=0).permute(0,3,1,2)
    flow_preds = flow_preds * mask
    return flow_preds