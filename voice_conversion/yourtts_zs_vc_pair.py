
"""
From the set of target wav files and a set of source files, it generates the correct speaker embeddings, 
then using a reference source file, it converts that file into the target speaker voice.

Usage:
    {0} [-v...] [options] 
    {0} -h | --help
    {0} --version
    
Options:
    -h --help                       Show this screen.
    --version                       Show version.
    -v, --verbose                   Increases the output verbosity level
    -o, --out-folder STR            Path to directory where the generated WAV files will be saved.
    -t, --target-files STR          Glob to the set of target files.
    -s, --source-files STR          Glob to the set of source files.
    -r, --reference-files STR       Glob to the reference source WAV files that will be converted.
    -i, --target-id STR             ID of the target subject. It is used in the generated name for reference.
"""


import sys
import glob
import os

import numpy as np

import pandas as pd

from docopt import docopt
from voice_conversion.yourtts_zs_vc_one import init_model, save_file


def run_one_sample(synthesizer, reference_emb, target_emb, target_id, ref_wav):    
    print(f"convert {ref_wav} to {target_id}")
    
    # reference_emb and target_emb are pandas Series, so we need to be careful
    waveform = synthesizer.one_voice_transfer(
            ref_wav=ref_wav,
            speaker_embedding=target_emb,     
            reference_speaker_embedding=reference_emb,
            speaker_id=None,
            reference_speaker_id=None,
    )
    return waveform


def main():
    """ """
    # Collect command line arguments
    # import ipdb; ipdb.set_trace()
    args = docopt(__doc__.format(sys.argv[0]), version="0.0.1")
    generated_folder = None
    if args['--out-folder']:
        generated_folder = args['--out-folder']
    target_files = args['--target-files']
    source_files = args['--source-files']
    ref_files = args['--reference-files']
    target_id= 'target'
    if args['--target-id']:
        target_id = args['--target-id']
    
    target_paths = glob.glob(target_files)
    source_paths = glob.glob(source_files)
    ref_paths = glob.glob(ref_files)
    
    synthesizer = init_model()
    
    target_emb = synthesizer.tts_model.speaker_manager.compute_embedding_from_clip(target_paths)
    source_emb = synthesizer.tts_model.speaker_manager.compute_embedding_from_clip(source_paths)

    for ref_file in ref_paths:
        ref_filename = os.path.basename(os.path.splitext(ref_file)[0])
        if not generated_folder or not os.path.exists(generated_folder):
            generated_folder = os.path.dirname(ref_file)
        generated_file = os.path.join(generated_folder, ref_filename+f'-to-{target_id}.wav')
        print(f"convert {ref_file} to {target_id}")
        waveform = synthesizer.one_voice_transfer(
                ref_wav=ref_file,
                speaker_embedding=target_emb,     
                reference_speaker_embedding=source_emb,
                speaker_id=None,
                reference_speaker_id=None,
        )
        save_file(synthesizer, waveform, generated_file)


if __name__ == "__main__":
    main()

