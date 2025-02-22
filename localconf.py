dependencies = ['torch', 'torchaudio', 'numpy']

import torch
import torch.nn.functional as F
import logging
import json
from pathlib import Path

from wavlm.WavLM import WavLM, WavLMConfig
from hifigan.models import Generator as HiFiGAN
from hifigan.utils import AttrDict
from matcher import KNeighborsVC

cp = Path(__file__).parent.absolute()


def knn_vc(pretrained=True, prematched=True, device='cuda') -> KNeighborsVC:
    """ Load kNN-VC (WavLM encoder and HiFiGAN decoder). Optionally use vocoder trained on `prematched` data. """
    hifigan, hifigan_cfg = hifigan_wavlm(pretrained, prematched, device)
    wavlm = wavlm_large(pretrained, device)
    knnvc = KNeighborsVC(wavlm, hifigan, hifigan_cfg, device)
    return knnvc


def hifigan_wavlm(pretrained=True, prematched=True, device='cuda') -> HiFiGAN:
    """ Load pretrained hifigan trained to vocode wavlm features. Optionally use weights trained on `prematched` data. """
    with open(cp/'hifigan'/'config_v1_wavlm.json') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    device = torch.device(device)

    generator = HiFiGAN(h).to(device)
    
    if pretrained:
        name = "g_02500000.pt"
        if prematched:
            name = 'prematch_' + name
        state_dict_g = torch.load(cp/'weights'/name)
        generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    print(f"[HiFiGAN] Generator loaded with {sum([p.numel() for p in generator.parameters()]):,d} parameters.")
    return generator, h


def wavlm_large(pretrained=True, device='cuda') -> WavLM:
    """Load the WavLM large checkpoint from the original paper. See https://github.com/microsoft/unilm/tree/master/wavlm for details. """
    if torch.cuda.is_available() == False:
        if str(device) != 'cpu':
            logging.warning(f"Overriding device {device} to cpu since no GPU is available.")
            device = 'cpu'
    checkpoint = torch.load(cp/'weights'/'WavLM-Large.pt')
    
    cfg = WavLMConfig(checkpoint['cfg'])
    device = torch.device(device)
    model = WavLM(cfg)
    if pretrained:
        model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    print(f"WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters.")
    return model
