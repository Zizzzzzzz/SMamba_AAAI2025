from omegaconf import DictConfig

from .SMamba import RNNDetector as SMambaRNNDetector


def build_recurrent_backbone(backbone_cfg: DictConfig):
    name = backbone_cfg.name
    if name == 'SMambaRNN':
        return SMambaRNNDetector(backbone_cfg)
    else:
        raise NotImplementedError