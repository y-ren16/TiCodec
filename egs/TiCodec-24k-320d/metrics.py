import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tqdm import tqdm
# from pystoi import stoi
# from pesq import pesq
import numpy as np
import csv
import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
import glob
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2
import librosa
import math

model_names = []

# model_names.append('logs_All_bl_1g2r')
# model_names.append('logs_All_conv_1g2r_8g3k1s_cos')

# model_names.append('logs_bl_LibriTTS_1')
# model_names.append('logs_convonly_Lib_1c_8g3k1s')
# model_names.append('logs_convonly_Lib_1c_8g3k1s_cosft')

# model_names.append('logs_bl_LibriTTS_1Group')
# model_names.append('logs_convonly_Lib_1group_2c_8g3k1s')
# model_names.append('logs_convonly_Lib_1group_2c_8g3k1s_compare_cos_ft')

# model_names.append('logs_convonly_Lib_1group_2c_8g3k1s_compare_cos')
# model_names.append('logs_convonly_Lib_1group_2c_8g3k1s_compare_mse')
# model_names.append('logs_convonly_Lib_1group_2c_8g3k1s_2inear')
# model_names.append('logs_convonly_1g2r_8g3k1s_256')

# model_names.append('logs_bl_LibriTTS_2Group')
# model_names.append('logs_convonly_Lib_2group_4c_8g3k1s')
# model_names.append('logs_convonly_Lib_2group_4c_8g3k1s_compare_cos')

# model_names.append('logs_convonly_Lib_1group_2c_8g4k2s')

# model_names.append('logs_bl_1g4r')
# model_names.append('logs_conv_1g4r')
# model_names.append('logs_encodec_320d_output_25_1')
# model_names.append('logs_encodec_320d_output_25_2')
# model_names.append('logs_encodec_320d_output_25_4')
# model_names.append('logs_soundstream_320d_output_25_1')
# model_names.append('logs_soundstream_320d_output_25_2')
# model_names.append('logs_soundstream_320d_output_25_4')

model_names.append('logs_convonly_Lib_1g1r_cos_from_head')
model_names.append('logs_convonly_Lib_1g2r_cos_from_head')
model_names.append('logs_convonly_Lib_1g4r_cos_from_head')

# writeheader_flag = {"LibriTTS": True, "AISHELL-3": True, "VCTK": True, "Musdb": True, "Audioset": True}
# writeheader_flag = {"LibriTTS": True, "AISHELL-3": True, "VCTK": True}
writeheader_flag = {"LibriTTS": True}
for model_name in model_names:
    print(model_name)
    print("=========================================")
    output_dir_path = os.path.join('../Paper_Data/GEN', model_name)
    # input_dirs = ['./Paper_Data/GT/LibriTTS', './Paper_Data/GT/AISHELL-3', './Paper_Data/GT/VCTK', './Paper_Data/GT/Musdb', './Paper_Data/GT/Audioset']
    # input_dirs = ['./Paper_Data/GT/LibriTTS', './Paper_Data/GT/AISHELL-3', './Paper_Data/GT/VCTK']
    input_dirs = ['../Paper_Data/GT/LibriTTS']
    # input_dirs = ['./Paper_Data/GT/AISHELL-3']
    # input_dirs = ['./Paper_Data/GT/VCTK']
    # input_dirs = ['./Paper_Data/GT/Musdb']
    # input_dirs = ['./Paper_Data/GT/Audioset']
    sample_rate=24000
    SISNR = ScaleInvariantSignalNoiseRatio()
    SISDR = ScaleInvariantSignalDistortionRatio()
    for input_dir in input_dirs:
        signal_all = []
        y_recons_all = []
        signal_name = []
        output_dir = output_dir_path + '/' + input_dir.split('/')[-1]
        print(output_dir)
        print(input_dir)
        input_files = glob.glob(f"{input_dir}/*.wav")
        input_files.sort()
        output_files = glob.glob(f"{output_dir}/*.wav")
        output_files.sort()
        assert len(input_files) == len(output_files)
        for i in tqdm(range(len(input_files))):
            input_file = input_files[i]
            output_file = output_files[i]
            assert os.path.basename(input_file) == os.path.basename(output_file)
            signal_name.append(os.path.basename(input_file))
            signal, signal_rate = librosa.load(
                input_file, sr=None
            )
            recons, recons_rate = librosa.load(
                output_file, sr=None
            )
            assert signal_rate == sample_rate
            assert recons_rate == sample_rate
            signal_all.append(signal)
            y_recons_all.append(recons)
        visqols = []
        pesq_scores = []
        stoi_scores = []
        sisnr = []
        sisdr = []
        mcd = []
        for i in tqdm(range(len(signal_all))):
            signal = signal_all[i]
            recons = y_recons_all[i]
            try:
                config = visqol_config_pb2.VisqolConfig()
                if input_dir.split('/')[-1] == "Musdb" or input_dir.split('/')[-1] == "Audioset":
                    mode == "audio"
                else:
                    mode = "speech"
                if mode == "audio":
                    target_sr = 48000
                    config.options.use_speech_scoring = False
                    svr_model_path = "libsvm_nu_svr_model.txt"
                elif mode == "speech":
                    target_sr = 16000
                    config.options.use_speech_scoring = True
                    svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
                else:
                    raise ValueError(f"Unrecognized mode: {mode}")
                config.audio.sample_rate = target_sr
                config.options.svr_model_path = os.path.join(
                    os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path
                )
                api = visqol_lib_py.VisqolApi()
                api.Create(config)
                ref = librosa.resample(y=signal, orig_sr=sample_rate, target_sr=target_sr)
                deg = librosa.resample(y=recons, orig_sr=sample_rate, target_sr=target_sr)
                _visqol = api.Measure(
                    ref.astype(float),
                    deg.astype(float),
                )
                visqols.append(_visqol.moslqo)
            except:
                print("visqol error!")
                print(signal_name[i])
                # visqols.append(min(visqols))

            try:
                ref = librosa.resample(y=signal, orig_sr=sample_rate, target_sr=16000)
                deg = librosa.resample(y=recons, orig_sr=sample_rate, target_sr=16000)
                min_len = min(len(ref), len(deg))
                ref = ref[:min_len]
                deg = deg[:min_len]
                # nb_pesq_scores = pesq(16000, ref, deg, 'nb')
                # wb_pesq_scores = pesq(16000, ref, deg, 'wb')
                wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
                wb_pesq_scores = wb_pesq(torch.from_numpy(deg), torch.from_numpy(ref)).numpy()
                pesq_scores.append(wb_pesq_scores)
            except:
                print("pesq error!")
                print(signal_name[i])
            try:
                ref = signal
                deg = recons
                min_len = min(len(ref), len(deg))
                ref = ref[:min_len]
                deg = deg[:min_len]
                # cur_stoi = stoi(ref, deg, sample_rate, extended=False)
                stoi = ShortTimeObjectiveIntelligibility(sample_rate, False)
                cur_stoi = stoi(torch.from_numpy(deg), torch.from_numpy(ref)).numpy()
                stoi_scores.append(cur_stoi)
            except:
                print("stoi error!")
                print(signal_name[i])

            try:
                ref = signal
                deg = recons
                min_len = min(len(ref), len(deg))
                ref = ref[:min_len]
                deg = deg[:min_len]
                sisnr.append(SISNR(torch.from_numpy(deg), torch.from_numpy(ref)).numpy())
                sisdr.append(SISDR(torch.from_numpy(deg), torch.from_numpy(ref)).numpy())

            except:
                print("sisnr error!")
                print(signal_name[i])

            try:
                ref = signal
                deg = recons
                min_len = min(len(ref), len(deg))
                ref = ref[:min_len]
                deg = deg[:min_len]
                org_mfcc = librosa.feature.melspectrogram(y=ref, sr=sample_rate,win_length=512,hop_length=128,n_fft=1024 ,n_mels=128,power=1.0).T
                target_mfcc = librosa.feature.melspectrogram(y=deg, sr=sample_rate,win_length=512,hop_length=128,n_fft=1024, n_mels=128,power=1.0).T
                mcd_data = 0.0
                for i in range(len(org_mfcc)):
                    diff = org_mfcc[i] - target_mfcc[i]
                    min_cost = 10 / math.log(10.0) * math.sqrt(2.0) * np.linalg.norm(diff)
                    mcd_data += np.mean(min_cost)
                mcd_data = mcd_data / len(org_mfcc)
                mcd.append(mcd_data)
            except:
                print("mcd error!")
                print(signal_name[i])

        print("visqol: ", np.mean(visqols))
        print("pesq: ", np.mean(pesq_scores))
        print("stoi: ", np.mean(stoi_scores))
        print("sisnr: ", np.mean(sisnr))
        print("sisdr: ", np.mean(sisdr))
        print("mcd: ", np.mean(mcd))
        visqol_mean = np.mean(visqols)
        pesq_mean = np.mean(pesq_scores)
        stoi_mean = np.mean(stoi_scores)
        sisnr_mean = np.mean(sisnr)
        sisdr_mean = np.mean(sisdr)
        mcd_mean = np.mean(mcd)
        print(f"{input_dir} finished!")
        print("=========================================")


        dataset_type = input_dir.split('/')[-1]
        with open(os.path.join('../Paper_Data/GEN',dataset_type + '.csv'), "a") as csvfile:
            keys = ["name", "visqol", "pesq", "stoi", "sisnr", "sisdr", "mcd"]
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            if writeheader_flag[dataset_type]:
                writer.writeheader()
                writeheader_flag[dataset_type] = False
            metadata = {
                "name": model_name+"_"+dataset_type,
                "visqol": str(visqol_mean),
                "pesq": str(pesq_mean),
                "stoi": str(stoi_mean),
                "sisnr": str(sisnr_mean),
                "sisdr": str(sisdr_mean),
                "mcd": str(mcd_mean),
            }
            writer.writerow(metadata)
