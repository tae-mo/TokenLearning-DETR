# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .token_ops import TokenFuser, TokenLearner

__all__ = [k for k in globals().keys() if not k.startswith("_")]

def build_model(args):
    return build(args)

