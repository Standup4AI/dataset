import pandas as pd
import glob
import os
from tqdm import tqdm
import numpy as np
def isNaN(num):
    return num != num

def df2json(df):
    """
    Transform a df of words into a format friendly to create seq labels.
    Works only for a df of 3 columns: text, t0, t1, t
    """
    words_chunks = []
    for line in df.values:
        text, t0, t1, _ = line
        words_chunks.append({'text': text, 'timestamp' : [t0, t1]})
    return words_chunks

def label_words_with_laughter(chunks, laughters, verbose=True):
    """
    laughters = {"0": {'start_sec': t0, 'end_sec' : t1}}
    chunks = [{'text': wd, 'timestamp': [t0, t1]}]
    """
    labels = ['O'] * len(chunks)  # Initialize all labels as 'O'

    # Iterate over each laughter
    for laughter in laughters.values():
        start = laughter['start_sec']
        end = laughter['end_sec']

        start_idx = None
        end_idx = None

        for i, word in enumerate(chunks):
            if isNaN(word['text']):  # Ugly kludge to deal with a past error, works for file en_us/oEc0UjgqtlI
                chunks[i]['text'] = 'None'
                if verbose: print("Added word 'None'..." )
            word_start, word_end = word['timestamp']
            word_start, word_end = float(word_start), float(word_end)
            
            # Heuristic for B
            prev_word_end = chunks[i - 1]['timestamp'][1] if i > 0 else 0
            if prev_word_end <= start <= word_start:
                start_idx = i - 1
            elif prev_word_end <= start <= word_end:
                start_idx = i - 1

            # Heuristic for L
            prev_word_end = chunks[i - 1]['timestamp'][1] if i > 0 else 0
            if prev_word_end <= end <= word_start:
                end_idx = i - 1
            elif prev_word_end <= end <= word_end:
                end_idx = i

        # Label words based on start and end indices
        if start_idx is not None and end_idx is not None:
            if start_idx == end_idx:
                labels[start_idx] = 'U'
            else:
                # if already in a serie # TO CHECK if correct
                if labels[start_idx] == 'L':
                    labels[start_idx] = 'I'
                else:
                    labels[start_idx] = 'B'
                labels[end_idx] = 'L'
                for i in range(start_idx + 1, end_idx):
                    labels[i] = 'I'

    # Combine the labels with text for output
    tagged_output = [
        {'text': chunk['text'].strip(), 'timestamp': chunk['timestamp'], 'label': label}
        for chunk, label in zip(chunks, labels)
    ]

    return tagged_output


def create_files_corpus(lang='es', ISpeech_only=True, Manual_annot=False):
    """
    Create the files corpus for the laughter detection.
    """

    # read csv of the transcript, which as col text, t0, t1, t
    list_transcript = [k.split('/')[-1].split('.')[0] for k in glob.glob(f"/home/user/data/standup/transcript/{lang}/mixed/*.csv")]

    for fn in tqdm(list_transcript):

        dfw = pd.read_csv(f"/home/user/data/standup/transcript/{lang}/mixed/{fn}.csv", sep=',')
        words_chunks = df2json(dfw)

        if Manual_annot:
            path_csv = f'/home/user/data/standup/dataset/test_laughters_manual_annotation/{lang}/{fn}.csv'
            if os.path.exists(path_csv):
                print(f"File found: {fn}.csv")
                dfl = pd.read_csv(path_csv)
                if len(dfl.columns) == 1:
                    dfl = pd.read_csv(path_csv, sep=';')
            else:
                continue
        else:
            if ISpeech_only:
                try:
                    dfl = pd.read_csv(f'/home/user/data/standup/laughter_detection/{lang}/Updated/{fn}.csv')
                except: 
                    print(f"File not found: {fn}.csv") # ugly kludge to deal with a past error: data/standup/laughter_detection/cs/Updated/6XMqZJbU0Rw.csv'
                    continue
                dfl = dfl[dfl.source == "Initial"]
            else:
                try:
                    # dfl = pd.read_csv(f'/home/user/data/standup/laughter_detection/{lang}/Updated_Filtered/{fn}.csv')
                    dfl = pd.read_csv(f'/home/user/data/standup/dataset/validaciones_nahuel/{lang}/{fn}.csv')
                except: 
                    print(f"File not found: {fn}.csv") # ugly kludge to deal with a past error: data/standup/laughter_detection/cs/Updated/6XMqZJbU0Rw.csv'
                    continue
                dfl = dfl[dfl.label == "risa"] # to change with the new output

        laughters = {i: {'start_sec': t0, 'end_sec': t1} for i, (t0, t1) in enumerate(dfl[['t0', 't1']].values)}
        tagged = label_words_with_laughter(words_chunks, laughters)

        if Manual_annot:
            dataset_type = 'Manual/test/'
        else:
            dataset_type = ISpeech_only*'IS' + (not ISpeech_only)*'Filtered' +'/all/'

        # save file
        path_csv = f'/home/user/data/standup/dataset/{lang}/' + dataset_type 
        if not os.path.exists(path_csv):
            os.makedirs(path_csv, exist_ok=True)

        pd.DataFrame(tagged).to_csv(path_csv + f'{fn}.csv', index=False)

    print (f'Language "{lang}" done!')

if __name__ == '__main__':

    fn = '_af-1wROuhw'
    dfw = pd.read_csv(f'/home/user/data/standup/transcript/test_fr/WXF_final_{fn}.csv',)
    words_chunks = df2json(dfw)

    dfl = pd.read_csv(f'/home/user/data/standup/laughter_detection/test_fr/Updated/{fn}.csv')

    laughters = {i: {'start_sec': t0, 'end_sec': t1} for i, (t0, t1) in enumerate(dfl[['t0', 't1']].values)}

    tagged = label_words_with_laughter(words_chunks, laughters)
    # To print tagged output:
    for word in tagged:
        print(f"{word['text']:>10} [{word['timestamp'][0]:.2f} - {word['timestamp'][1]:.2f}]: {word['label']}")