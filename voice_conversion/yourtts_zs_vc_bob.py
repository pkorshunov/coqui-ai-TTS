
"""
From the set of target wav files and a set of source files, it generates the correct speaker embeddings, 
then using a reference source file, it converts that file into the target speaker voice.

Usage:
    {0} [-v...] [options] <real_probes>
    {0} -h | --help
    {0} --version
Arguments:
    <real_probes>                   CSV file list of Bob database with real probes.
    
Options:
    -h --help                       Show this screen.
    --version                       Show version.
    -v, --verbose                   Increases the output verbosity level
    -o, --out-folder STR            Path to directory where the generated WAV files will be saved.
                                    [default: ./data].
    -d, --prefix-datafolder STR     Path to directory where the original WAV files from <real_probes> are.
                                    [default: ./data].
    -n, --num-utt-adapt INT         Number of samples used to compute speaker embeddgins.
                                    [default: 20].
    -t, --adapt-test-size INT       Number of samples for which we generate fake data.
                                    [default: 10].
    -p, --vc-pairs INT              Number of voice conversion pairs for each speakers.
                                    [default: 5].
"""


import sys
import glob
import os

import numpy as np

import pandas as pd

from docopt import docopt
from voice_conversion.yourtts_zs_vc_one import init_model, save_file

# # real_file = '/remote/idiap.svm/temp.biometric02/pkorshunov/src/natspeech/data/lists/libritts/real/dev/for_probes.csv'

def run_one_sample(synthesizer, prefix_folder, reference_emb, ref_id, target_emb, target_id, ref_wav):
    ref_wav = os.path.join(prefix_folder, ref_wav+'.wav')
    
    print(f"convert {ref_wav} of {ref_id} to {target_id}")
    
    # reference_emb and target_emb are pandas Series, so we need to be careful
    waveform = synthesizer.one_voice_transfer(
            ref_wav=ref_wav,
            speaker_embedding=target_emb.iloc[0],     
            reference_speaker_embedding=reference_emb.iloc[0],
            speaker_id=None,
            reference_speaker_id=None,
    )
    return waveform

def compute_embedding(synthesizer, prefix_folder, file_list):
    file_paths = [os.path.join(prefix_folder, f+'.wav') for f in file_list]
    print(f"Compute embeddging for {file_paths[0]}")

    emd = synthesizer.tts_model.speaker_manager.compute_embedding_from_clip(file_paths)
    return emd

def main():
    """ """
    # Collect command line arguments
    args = docopt(__doc__.format(sys.argv[0]), version="0.0.1")
    probes_filelist = args['<real_probes>']
    generated_folder = args['--out-folder']
    prefix_folder = args['--prefix-datafolder']
    num_utt_embeddings = int(args['--num-utt-adapt'])
    num_test_size = int(args['--adapt-test-size'])
    num_vc_pairs = int(args['--vc-pairs'])
    
    synthesizer = init_model()
    
    real_data = pd.read_csv(probes_filelist, na_values='nan')
    sample_paths = real_data.groupby('REFERENCE_ID').sample(n=num_utt_embeddings, random_state=2022)
    embeddings_data = sample_paths.groupby('REFERENCE_ID')['PATH'].apply(list).reset_index(name='paths_list')
    embeddings_data['embeddings'] = embeddings_data['paths_list'].apply(lambda file_list: 
        compute_embedding(synthesizer=synthesizer, prefix_folder=prefix_folder, file_list=file_list)
    )

    rng = np.random.default_rng(2022)
    
    list_of_ids = real_data.REFERENCE_ID.unique()
    # sampled_id_paths = real_data.groupby('REFERENCE_ID').sample(n=num_test_size, random_state=2022)
    sampled_id_paths = sample_paths.groupby('REFERENCE_ID').sample(n=num_test_size, random_state=2022)
    
    generated_list = list()
    for cur_ref_id, cur_ref_id_group in sampled_id_paths.groupby('REFERENCE_ID'):
        list_excl_cur_id = np.delete(list_of_ids, list_of_ids==cur_ref_id)
        # ensure that we do not have repetitions inside the randomly sampled list of num_vc_pairs elements
        list_excl_cur_id = rng.choice(list_excl_cur_id, num_vc_pairs, replace=False, shuffle=False)
        reference_emb = embeddings_data[embeddings_data['REFERENCE_ID'] == cur_ref_id]['embeddings']
        for target_id in list_excl_cur_id:
            target_emb = embeddings_data[embeddings_data['REFERENCE_ID'] == target_id]['embeddings']
            for row_index, cur_row in cur_ref_id_group.iterrows():
                ref_wav = cur_row['PATH']
                generated_id = ref_wav.split('/')[-1] + '_to_' + str(target_id)
                generated_file = os.path.join(generated_folder, generated_id)
                wav = run_one_sample(synthesizer, prefix_folder, reference_emb, cur_ref_id, target_emb, target_id, ref_wav)
                save_file(synthesizer, wav, generated_file + '.wav')
                generated_list.append((generated_file, target_id, generated_id))

    generated_data = pd.DataFrame(generated_list, columns=['PATH', 'REFERENCE_ID', 'ID'])
    generated_data.sort_values(by=['REFERENCE_ID', 'PATH']).to_csv(os.path.join(generated_folder, 'for_probes.csv'), index=False)
    
    
if __name__ == "__main__":
    main()

