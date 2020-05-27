# overview
This is a repository for technical test of various ambisonics system.

# copyright
all right reserved to deENTA,Inc.

# Auther
Kenta KANASAKI<kenta@enta.tokyo>

# dependencies

# How to install

# Example
Execute at repository root.
## play with 4ch wav file and std audio out.
python src/test.py -d=1 -s=/dev/sd1/wav/0001.wav
[1] program source
[2] audio dev number
[3] path to audio file

## play with 4ch wav m3u8 local streaming
python src/test-hls.py -mode=hls_local -d=1 -s=/Document/public/streaming.m3u8

## play with 4ch wav m3u8 network steaming
python src/test-hls.py -mode=hls_network -d=1 -s=https://dev.enta.tokyo/ambi-test-0001.m3u8


