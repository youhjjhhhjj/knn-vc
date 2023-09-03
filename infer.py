
from pathlib import Path
import argparse
import os
import torchaudio
from localconf import knn_vc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', help='Path to directory containing all source waveforms to be converted', type=Path, required=True)
    parser.add_argument('-ref', help='Path to directory containing all reference waveforms from the target speaker', type=Path, required=True)
    parser.add_argument('-out', help='Path to directory where output will be saved', type=Path, default=Path(__file__).parent/'out')
    args = parser.parse_args()

    assert args.src.is_dir()
    assert args.ref.is_dir()
    assert args.out.is_dir()

    model = knn_vc()
    src_path: Path = args.src.resolve()
    ref_path: Path = args.ref.resolve()
    refs = []
    for f in os.listdir(ref_path):
        if os.path.isfile(ref_path/f):
            refs.append(ref_path/f)
    matching_set = model.get_matching_set(refs)

    for f in os.listdir(src_path):
        if os.path.isfile(src_path/f):
            out_wav = model.match(model.get_features(src_path/f), matching_set, topk=4)
            torchaudio.save(args.out.resolve()/Path(f).with_suffix(".wav").name, out_wav[None], 16000)
            print("finished converting " + f)