# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
from ...util.misc import unnormalize_parameter

# from .DABDETR import sigmoid_focal_loss
from util import box_ops


def prepare_for_cdn(
    dn_args,
    training,
    num_queries,
    num_classes,
    hidden_dim,
    label_enc,
):
    """
    To the DINO denoising implementation, we add noise over the primitive parameters and not only the boxes
    ------------------------------------------------------------------------
    A major difference of DINO from DN-DETR is that the author process pattern embedding in its detector
    forward function and use learnable tgt embedding, so we change this function a little bit.
    :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
    :param training: if it is training or inference
    :param num_queries: number of queries
    :param num_classes: number of classes
    :param hidden_dim: transformer hidden dim
    :param label_enc: encode labels in dn
    :return:
    """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t["labels"])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        unmask_param = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        parameters = torch.cat([t["parameters"] for t in targets])
        batch_idx = torch.cat(
            [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )

        known_indice = torch.nonzero(unmask_label + unmask_param)
        known_indice = known_indice.view(-1)
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)

        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_params = parameters.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_param_expand = known_params.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, num_classes)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = (
            torch.tensor(range(len(parameters)))
            .long()
            .cuda()
            .unsqueeze(0)
            .repeat(dn_number, 1)
        )
        positive_idx += (
            (torch.tensor(range(dn_number)) * len(parameters) * 2)
            .long()
            .cuda()
            .unsqueeze(1)
        )
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(parameters)
        known_bboxs = known_params[:, 14:18]
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2  # w/2,h/2
            diff[:, 2:] = known_bboxs[:, 2:] / 2  # w/2,h/2

            diff_list = [diff] * 3 + [diff[:, 2:4]] + [diff]
            diff = torch.cat(diff_list, dim=1)

            rand_sign = (
                torch.randint_like(known_params, low=0, high=2, dtype=torch.float32)
                * 2.0
                - 1.0
            )

            rand_part = torch.rand_like(known_params)
            rand_part[negative_idx] += 1.0

            rand_part *= rand_sign
            known_param_xyxy = box_ops.param_cxcywh_to_xyxy(known_params)
            known_param_ = (
                known_param_xyxy + torch.mul(rand_part, diff).cuda() * box_noise_scale
            )
            known_param_ = known_param_.clamp(min=0.0, max=1.0)  # xyxy
            known_param_expand = box_ops.param_xyxy_to_cxcywh(known_param_)

        m = known_labels_expaned.long().to("cuda")
        input_label_embed = label_enc(m)
        input_param_embed = unnormalize_parameter(known_param_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_param = torch.zeros(pad_size, 18).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_param = padding_param.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )  # [1,2, 1,2,3]
            map_known_indice = torch.cat(
                [map_known_indice + single_pad * i for i in range(2 * dn_number)]
            ).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_param[(known_bid.long(), map_known_indice)] = input_param_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1),
                    single_pad * 2 * (i + 1) : pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * i * 2
                ] = True
            else:
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1),
                    single_pad * 2 * (i + 1) : pad_size,
                ] = True
                attn_mask[
                    single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * 2 * i
                ] = True
        dn_meta = {
            "pad_size": pad_size,
            "num_dn_group": dn_number,
            # "positive_idx": positive_idx,
            # "negative_idx": negative_idx,
            # "rand_part": torch.mul(rand_part, diff).cuda() * box_noise_scale,
        }

    else:
        input_query_label = None
        input_query_param = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_param, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
    post process of dn after output from the transformer
    put the dn part in the dn_meta
    """
    if dn_meta and dn_meta["pad_size"] > 0:
        output_known_class = outputs_class[:, :, : dn_meta["pad_size"], :]
        output_known_coord = outputs_coord[:, :, : dn_meta["pad_size"], :]
        outputs_class = outputs_class[:, :, dn_meta["pad_size"] :, :]
        outputs_coord = outputs_coord[:, :, dn_meta["pad_size"] :, :]
        out = {
            "pred_logits": output_known_class[-1],
            "pred_params": output_known_coord[-1],
        }
        if aux_loss:
            out["aux_outputs"] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta["output_known_lbs_params"] = out
    return outputs_class, outputs_coord
