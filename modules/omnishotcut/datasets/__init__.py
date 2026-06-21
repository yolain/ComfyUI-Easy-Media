# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
def build_dataset(set_type, args):
    # set_type refers to using Training Dataset or Testing Dataset

    if args.dataset_file == 'shot_boundary_detection':
        from .dataloader import build as build_cut_anything

        return build_cut_anything(args, set_type)

    raise ValueError(f'dataset {args.dataset_file} not supported')
