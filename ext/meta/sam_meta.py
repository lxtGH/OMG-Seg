meta_dict = {
    'vit_h': dict(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        # common
        prompt_embed_dim=256,
        image_size=1024,
        vit_patch_size=16,
        image_embedding_size=64
    ),
    'vit_l': dict(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        # common
        prompt_embed_dim=256,
        image_size=1024,
        vit_patch_size=16,
        image_embedding_size=64
    ),
    'vit_b': dict(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # common
        prompt_embed_dim=256,
        image_size=1024,
        vit_patch_size=16,
        image_embedding_size=64
    )
}

checkpoint_dict = {
    'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
}
