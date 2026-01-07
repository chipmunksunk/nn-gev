import argparse
import os
import librosa
import glob
import numpy as np

from fgnt.utils import Timer, mkdir_p
from fgnt.signal_processing import audiowrite, stft, istft, istft_mc, compound_normalization
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

# Create output directory if it does not exist
mkdir_p(args.output_dir)

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
    S_data, _ = librosa.load(os.path.join(path_audio, "setup_"+str(i+1)+"_speech.wav"), mono=False, sr=None)
    N_data, _ = librosa.load(os.path.join(path_audio, "setup_"+str(i+1)+"_noise.wav"), mono=False, sr=None)

    S = stft(S_data, size=1024, shift=512, time_dim=1).transpose((1, 0, 2)) #(nFrames, nChannels, nFreqs)
    N = stft(N_data, size=1024, shift=512, time_dim=1).transpose((1, 0, 2))
    Y = S + N

    Y_var = Variable(np.abs(Y).astype(np.float32))
    if args.gpu >= 0:
        Y_var.to_gpu(args.gpu)
    with Timer() as t:
        N_masks, X_masks = model.calc_masks(Y_var)
        N_masks.to_cpu()
        X_masks.to_cpu()
    t_net += t.msecs

    with Timer() as t:
        N_mask = np.median(N_masks.data, axis=1) #(nFrames, nFreqs): Median over channels
        X_mask = np.median(X_masks.data, axis=1)
        S_hat, N_hat = gev_wrapper_on_masks(S, N, N_mask, X_mask)
        Y_hat = S_hat + N_hat
    t_beamform += t.msecs

    with Timer() as t:
        s_hat = istft_mc(S_hat, size=1024, shift=512) #(nSamples, nChannels)
        n_hat = istft_mc(N_hat, size=1024, shift=512)
        y_hat = istft_mc(Y_hat, size=1024, shift=512)

        #Only normalize y_hat within audiowrite. S_hat and N_hat is normalized within compound_normalization and should retain scaling.
        s_hat, n_hat = compound_normalization(s_hat, n_hat)
        audiowrite(s_hat, os.path.join(args.output_dir, f'setup_{i+1}_d_hat.wav'), 16000, normalize=False, threaded=True)
        audiowrite(n_hat, os.path.join(args.output_dir, f'setup_{i+1}_noise_hat.wav'), 16000, normalize=False, threaded=True)
        audiowrite(y_hat, os.path.join(args.output_dir, f'setup_{i+1}_total_hat.wav'), 16000, normalize=True, threaded=True)
    t_io += t.msecs

print('Finished')
print('Timings: I/O: {:.2f}s | Net: {:.2f}s | Beamformer: {:.2f}s'.format(
        t_io / 1000, t_net / 1000, t_beamform / 1000
))