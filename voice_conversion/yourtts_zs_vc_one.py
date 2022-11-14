"""
The script is adapted from Zero-Shot VC demo shown here: https://github.com/Edresson/YourTTS/
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
    -t, --target-files STR          Glob to the set of target files.
    -s, --source-files STR          Glob to the set of source files.
    -r, --reference-file STR        Path to the reference source WAV file that will be converted.
    -o, --out-file STR              Path where the generated WAV file will be saved.
"""

import sys
import os
import glob

from docopt import docopt

import string
import time
import argparse
import json

import numpy as np

import torch

from pathlib import Path
from TTS.utils.manage import ModelManager

from TTS.utils.synthesizer import Synthesizer

# model vars
MODEL_NAME = 'tts_models/multilingual/multi-dataset/your_tts'
# MODEL_PATH = 'best_model.pth.tar'
# CONFIG_PATH = 'config.json'
# TTS_LANGUAGES = "language_ids.json"
# TTS_SPEAKERS = "speakers.json"
USE_CUDA = torch.cuda.is_available()

def init_model():
    path = Path(__file__).parent / "../TTS/.models.json"
    manager = ModelManager(path)
    model_path, config_path, model_item = manager.download_model(MODEL_NAME)
    # import ipdb; ipdb.set_trace()
    # vocoder_name = model_item["default_vocoder"]
    # vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)
    
    synthesizer = Synthesizer(
        model_path,
        config_path,
        tts_speakers_file="",
        tts_languages_file="",
        vocoder_checkpoint=None,
        vocoder_config=None,
        encoder_checkpoint = None,
        encoder_config = None,
        use_cuda = USE_CUDA,
    )
    return synthesizer
    
def save_file(synthesizer, wav, out_file):
    # save the results
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    print(" > Saving output to {}".format(out_file))
    synthesizer.save_wav(wav, out_file)
    
def main():
    """ """
    # Collect command line arguments
    args = docopt(__doc__.format(sys.argv[0]), version="0.0.1")
    target_files = args['--target-files']
    source_files = args['--source-files']
    ref_file = args['--reference-file']
    out_file = args['--out-file']
    
    target_paths = glob.glob(target_files)
    source_paths = glob.glob(source_files)
    
    synthesizer = init_model()
    
    wavs = synthesizer.tts(
        text="",
        speaker_name="",
        language_name='en',
        speaker_wav=target_paths,
        reference_wav=ref_file,
        style_wav=None,
        style_text=None,
        reference_speaker_name=None,
        source_wav=source_paths,
    )

    save_file(synthesizer, wavs[0], out_file)
    
    
if __name__ == "__main__":
    main()

