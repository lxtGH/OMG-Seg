import torch


def preprocess_video_panoptic_gt(
        gt_labels,
        gt_masks,
        gt_semantic_seg,
        gt_instance_ids,
        num_things,
        num_stuff,
):
    num_classes = num_things + num_stuff
    num_frames = len(gt_masks)
    mask_size = gt_masks[0].masks.shape[-2:]

    thing_masks_list = []
    for frame_id in range(num_frames):
        thing_masks_list.append(gt_masks[frame_id].pad(
            mask_size, pad_val=0).to_tensor(
            dtype=torch.bool, device=gt_labels.device)
        )
    instances = torch.unique(gt_instance_ids[:, 1])
    things_masks = []
    labels = []
    for instance in instances:
        pos_ins = torch.nonzero(torch.eq(gt_instance_ids[:, 1], instance), as_tuple=True)[0]  # 0 is for redundant tuple
        labels_instance = gt_labels[:, 1][pos_ins]
        assert torch.allclose(labels_instance, labels_instance[0])
        labels.append(labels_instance[0])
        instance_frame_ids = gt_instance_ids[:, 0][pos_ins].to(dtype=torch.int32).tolist()
        instance_masks = []
        for frame_id in range(num_frames):
            frame_instance_ids = gt_instance_ids[gt_instance_ids[:, 0] == frame_id, 1]
            if frame_id not in instance_frame_ids:
                empty_mask = torch.zeros(
                    mask_size,
                    dtype=thing_masks_list[frame_id].dtype, device=thing_masks_list[frame_id].device
                )
                instance_masks.append(empty_mask)
            else:
                pos_inner_frame = torch.nonzero(torch.eq(frame_instance_ids, instance), as_tuple=True)[0].item()
                frame_mask = thing_masks_list[frame_id][pos_inner_frame]
                instance_masks.append(frame_mask)
        things_masks.append(torch.stack(instance_masks))

    if len(instances) == 0:
        things_masks = torch.stack(thing_masks_list, dim=1)
        labels = torch.empty_like(instances)
    else:
        things_masks = torch.stack(things_masks)
        labels = torch.stack(labels)
    assert torch.all(torch.less(labels, num_things))

    if gt_semantic_seg is not None:
        things_labels = labels
        gt_semantic_seg = gt_semantic_seg.squeeze(1)

        semantic_labels = torch.unique(
            gt_semantic_seg,
            sorted=False,
            return_inverse=False,
            return_counts=False)
        stuff_masks_list = []
        stuff_labels_list = []
        for label in semantic_labels:
            if label < num_things or label >= num_classes:
                continue
            stuff_mask = gt_semantic_seg == label
            stuff_masks_list.append(stuff_mask)
            stuff_labels_list.append(label)

        if len(stuff_masks_list) > 0:
            stuff_masks = torch.stack(stuff_masks_list, dim=0)
            stuff_labels = torch.stack(stuff_labels_list, dim=0)
            assert torch.all(torch.ge(stuff_labels, num_things)) and torch.all(torch.less(stuff_labels, num_classes))
            labels = torch.cat([things_labels, stuff_labels], dim=0)
            masks = torch.cat([things_masks, stuff_masks], dim=0)
        else:
            labels = things_labels
            masks = things_masks
        assert len(labels) == len(masks)
    else:
        masks = things_masks

    labels = labels.to(dtype=torch.long)
    masks = masks.to(dtype=torch.long)
    return labels, masks
