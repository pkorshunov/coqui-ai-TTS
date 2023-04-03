#!/bin/bash

flist=$1
prefixdir=$2

target_wavs="$prefixdir/raw/swan/4_00002_f_01_01*_t_2.wav"
source_wavs="$prefixdir/raw/swan//4_00004_f_01_01*_t_2.wav"

#for f in "${@}"; do
while IFS="" read -r p || [ -n "$p" ]; do
    reffile="${prefixdir}/raw/swan/${p}"
    outfile="${prefixdir}/vc/swan/${p}"
    echo "Converting $reffile into $outfile"
    python voice_conversion/yourtts_zs_vc_one.py --reference-file $reffile --target-files $target_wavs --source-files $source_wavs --out-file $outfile
done < $flist
