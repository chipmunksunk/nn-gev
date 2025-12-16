import argparse
import os
import librosa
import glob
import numpy as np

from fgnt.utils import Timer
from fgnt.signal_processing import audiowrite, stft, istft
from fgnt.beamforming import gev_wrapper_on_masks

import chainer
from chainer import Variable
from chainer import cuda
from chainer import serializers
from nn_models import BLSTMMaskEstimator, SimpleFWMaskEstimator

parser = argparse.ArgumentParser(description='NN GEV beamforming')
parser.add_argument('audio_data_path',
                    help='Path to directory containing audio data to process in form of wav files.')
parser.add_argument('output_dir',
                    help='The directory where the enhanced wav files will '
                         'be stored.')
parser.add_argument('model',
                    help='Trained model file')
parser.add_argument('model_type',
                    help='Type of model (BLSTM or FW)')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# Prepare model
if args.model_type == 'BLSTM':
    model = BLSTMMaskEstimator()
elif args.model_type == 'FW':
    model = SimpleFWMaskEstimator()
else:
    raise ValueError('Unknown model type. Possible are "BLSTM" and "FW"')

serializers.load_hdf5(args.model, model)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy
chainer.no_backprop_mode()

t_io = 0
t_net = 0
t_beamform = 0

path_audio = args.audio_data_path
nSetups = len(glob.glob(os.path.join(path_audio, 'setup_*_clean_speech.wav')))

# Beamform loop
for i in range(nSetups):
    audio_data, _ = librosa.load(os.path.join(path_audio, "setup_"+str(i+1)+"_total.wav"), mono=False, sr=None)

    Y = stft(audio_data, size=1024, shift=512, time_dim=1).transpose((1, 0, 2))
    Y_var = Variable(np.abs(Y).astype(np.float32))
    if args.gpu >= 0:
        Y_var.to_gpu(args.gpu)
    with Timer() as t:
        N_masks, X_masks = model.calc_masks(Y_var)
        N_masks.to_cpu()
        X_masks.to_cpu()
    t_net += t.msecs

    with Timer() as t:
        N_mask = np.median(N_masks.data, axis=1)
        X_mask = np.median(X_masks.data, axis=1)
        Y_hat = gev_wrapper_on_masks(Y, N_mask, X_mask)
    t_beamform += t.msecs

    #adjust this to write to the provided output directory
    filename = os.path.join(
            args.output_dir,
            'debug_output.wav'
    )

    with Timer() as t:
        audiowrite(istft(Y_hat, size=1024, shift=512), filename, 16000, True, True)
    t_io += t.msecs

print('Finished')
print('Timings: I/O: {:.2f}s | Net: {:.2f}s | Beamformer: {:.2f}s'.format(
        t_io / 1000, t_net / 1000, t_beamform / 1000
))