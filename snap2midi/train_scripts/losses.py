import torch
import torch.nn.functional as F

def bce(output, target, mask):
    """
        Binary crossentropy with masking.

        Args:
            output (torch.Tensor): Model's output
            target (torch.Tensor): Target tensor
            mask (torch.Tensor): Mask
        
        Returns:
            BCE Loss
            
    """
    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    return torch.sum(matrix * mask) / torch.sum(mask)


def regress_bce(output_dict, target_dict):
    """High-resolution piano note regression loss, including onset regression, 
    offset regression, velocity regression and frame-wise classification losses.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['label_reg_onsets'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['label_reg_offsets'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['label_frames'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['label_velocities'] / 128, target_dict['label_onsets'])
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss

    loss_dict = {
        'onset_loss': onset_loss,
        'offset_loss': offset_loss,
        'frame_loss': frame_loss,
        'total_loss': total_loss,
        'velocity_loss': velocity_loss
    }

    return loss_dict


def regress_pedal_bce(output_dict, target_dict):
    """High-resolution piano pedal regression loss, including pedal onset 
    regression, pedal offset regression and pedal frame-wise classification losses.
    """
    onset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_onset_output'], target_dict['pedal_reg_onset'][:, :, None])
    offset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_offset_output'], target_dict['pedal_reg_offset'][:, :, None])
    frame_pedal_loss = F.binary_cross_entropy(output_dict['pedal_frame_output'], target_dict['pedal_frames'][:, :, None])
    total_loss = onset_pedal_loss + offset_pedal_loss + frame_pedal_loss
    return total_loss