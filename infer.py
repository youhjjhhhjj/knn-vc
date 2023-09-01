
from pathlib import Path
import argparse
import os
import torchaudio
from localconf import knn_vc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', help='Path to source waveform', type=Path, required=True)
    parser.add_argument('-ref', help='Path to directory containing all reference waveforms from the target speaker', type=Path, required=True)
    parser.add_argument('-out', help='Path to directory of output', type=Path, default=Path(__file__).parent/'out')
    args = parser.parse_args()

    assert args.src.is_file()
    assert args.ref.is_dir()
    assert args.out.is_dir()

    model = knn_vc()
    query_seq = model.get_features(args.src.resolve())
    ref_path = args.ref.resolve()
    refs = []
    for f in os.listdir(ref_path):
        if os.path.isfile(ref_path/f):
            refs.append(ref_path/f)
    matching_set = model.get_matching_set(refs)

    out_wav = model.match(query_seq, matching_set, topk=4)
    torchaudio.save(args.out.absolute()/'knnvc1_out.wav', out_wav[None], 16000)