import os
import glob
import whisperx
from utils import whisperX2csv

def whisperX_main(list_files_index = ['_af-1wROuhw'], # list of indexes
                  path_wav = '/home/user/data/standup/audio/test_fr/',
                  path_csv = '/home/user/data/standup/transcript/test_fr/whisperx-large-v3-french/',
                  model_name = "bofenghuang/whisper-large-v3-french", # can use "large-v3", or any model converted previsouly to fast-whisper format
                  lang=None,
                  device = "cuda",
                  batch_size = 32, # reduce if low on GPU mem
                  compute_type = "float16", # change to "int8" if low on GPU mem (may reduce accuracy)
                  ):
    
    # Create output folder if not existing
    os.makedirs(path_csv, exist_ok=True)

    if lang == None:
        lang = path_wav.split('/')[-2].split('_')[0] # do not work for test_fr though...
        print(f'Automatically adding lang as: {lang}')

    # this was used to debug, not sure it is useful. Just converting to fast whisper seems to solve the prob
    # from huggingface_hub import snapshot_download
    # model_path = snapshot_download(model_name)
    model = whisperx.load_model(model_name, device, compute_type=compute_type)

    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

    for fn in list_files_index:
        if not os.path.isfile(path_csv + f'{fn}.csv'):
            audio_file = path_wav + f'{fn}.wav'

            audio = whisperx.load_audio(audio_file)
            result = model.transcribe(audio, batch_size=batch_size)
            # print(result["segments"]) # before alignment
            if result["language"] == lang:
                model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
                result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

                df = whisperX2csv(result)
                df.to_csv(path_csv + f'{fn}.csv')
            else:
                print(f'Issue with file {fn}... language {result["language"]} detected')
                model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
                result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

                df = whisperX2csv(result)
                df.to_csv(path_csv + f'{fn}.csv')
                with open(path_csv + 'errors_lang.txt', 'a') as f:
                    f.write(f'{fn}\n')

        else:
            print(f'Skip file... {path_csv}{fn}.csv already existing')

if __name__ == '__main__':

    path_data_standup = '/home/user/data/standup/'

    lang = 'test_fr'
    
    path_wav = path_data_standup + f'audio/{lang}/'
    list_files_index = [os.path.split(k)[1].split('.')[0] for k in glob.glob(path_wav + "*.wav")]
    whisperX_main(list_files_index = list_files_index, # list of indexes
                path_wav = path_wav,
                path_csv = path_data_standup + f'transcript/{lang}/whisperx-large-v3-french/',
                model_name = "/home/user/standup_comedy/bofenghuang/faster-whisper-large-v3-french", # can use "large-v3", or any model converted previously to fast-whisper format, see https://github.com/SYSTRAN/faster-whisper#model-conversion
                device = "cuda",
                batch_size = 32, # reduce if low on GPU mem
                compute_type = "float16", # change to "int8" if low on GPU mem (may reduce accuracy)
                )