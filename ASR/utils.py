import pandas as pd
import os
import glob
# import gc 

def ASRNormalization(df):
    df[['t0', 't1']] = pd.DataFrame(df['timestamp'].to_list(), index=df.index)
    # Drop the original timestamp column
    df = df.drop(columns='timestamp')
    df['t'] = df['t1'] - df['t0']
    return df 

def whisperjson2csv(path_json = 'home/user/data/standup/transcript/test_fr/seg_whisper-large-v3/_af-1wROuhw.json'):
    """
    Files originally obtained from whisper
    """
    df = pd.read_json(path_json)['chunks'].apply(pd.Series)
    return ASRNormalization(df) 
     #df.to_csv('~/data/standup/transcript/test_fr/W__af-1wROuhw.csv')

def whisperX2csv(result):
    """
    Files originally obtained from whisperX
    """
    # normalizing the files to csv
    words_chunks = []
    for k in result['word_segments']: 
        if 'start' in k.keys():
            words_chunks.append({'text' : k['word'], 'timestamp' : [k['start'], k['end']], 'score' : k['score']})
        else:
            if len(words_chunks):
                words_chunks[-1]['text'] += ' '+k['word']
            else:
                print("Issue with the first word")

    df = pd.DataFrame(words_chunks)
    return ASRNormalization(df) 

def fix_timestamps(dfx, 
                   dfw,
                   thresh_long_words_x = 1.0,
                   thresh_long_words_whisper = 0.8,
                   thresh_laughter_end=0.1, # time to add before the end of laughters and beginning of the future word
                   thresh_laughter_begin=0.01, # time to add before the beginning of laughters and end of the last word 
                   verbose=False,
                   ):
    """
    Mix the outputs of whisperX and whisper helps to fix word timestamps next to laughters, and detect more of them. 
    Whisper groups the laughter with the words after them, and whisperX before them. 
    Using this, it is possible to get the right timestamps, and detect new ones that are the space in between the words.   
    """

    dfx.drop('score', axis=1, inplace=True) # not sure to do this, but take away the score column

    dfbx = dfx[dfx.t > thresh_long_words_x]
    dfbf = dfw[dfw.t > thresh_long_words_whisper]
    
    new_laughters = []

    for idx in dfbx.index:
        wd, t0, t1, _ = dfbx.loc[idx].values
        # long wf word comes during a long wx word  
        dfinter = dfbf[(dfbf.t0 > t0) & (dfbf.t0 < t1)]

        if len(dfinter) == 0:
            if verbose: print('error dfbx', t0, t1, len(dfinter))
        else:
            if len(dfinter) == 1:
                _, t0f, t1f, _ = dfinter.values[0]
                dfx.loc[idx] = wd, t0, t0f, t0f - t0
                if verbose: print('changed', t0, t1, t0f)
            else:
                _, t0f, t1f, _ = dfinter.iloc[0].values
                dfx.loc[idx] = wd, t0, t0f, t0f - t0
                if verbose: print('changed', t0, t1, t0f)
            
            new_laughters.append([t0f+thresh_laughter_begin, t1-thresh_laughter_end]) # can be problematic if intersection between long words is less than 0.1s
            
    if verbose: print(new_laughters)

    return dfx, new_laughters

def add_new_laughters(laughters_base, 
                      new_laughters, 
                      t_min):
    """
    Add the new laughters to the originally detected ones. 
    Only the new laughters with no intersection with an existing laughter are kept.
    t_min sec minimum
    """
    list_add_laughters = []
    for new_laughter in new_laughters:
        t0, t1 = new_laughter
        # if longer than t_min
        if t1 - t0 > t_min:
            bool_intersection = False
            for laughter in laughters_base:
                if t0 < laughter[0]:
                    if t1 > laughter[0]:
                        bool_intersection = True 
                else:
                    if t0 < laughter[1]:
                        bool_intersection = True 
            if not bool_intersection:
                list_add_laughters.append(new_laughter)

    return list_add_laughters

def LaughtersNormalization(laughters_base, new_laughters, t_min=0.):
    """
    Add the laughters detected from the whisper/whisperX method
    Output a DataFrame
    """
    list_add_laughters = add_new_laughters(laughters_base, new_laughters, t_min)  
    df = pd.DataFrame(list(laughters_base) + list_add_laughters, columns=['t0', 't1'])
    df['source'] = ['Initial']*len(laughters_base) + ['Added']*len(list_add_laughters)
    df.sort_values(by='t0', inplace=True) # sort the df regarding t0   
    return df

if __name__ == '__main__':

    path_data_standup = '/home/user/data/standup/'
    fn = '_af-1wROuhw'
    dfx = pd.read_csv(path_data_standup + f'transcript/test_fr/WX_{fn}.csv', index_col=0)
    dfw = pd.read_csv(path_data_standup + f'transcript/test_fr/WF_{fn}.csv', index_col=0)

    # fixing timestamps errors due to laughters or "voice noise"
    dfx, new_laughters = fix_timestamps(dfx, dfw)
    dfx.to_csv(path_data_standup + f'transcript/test_fr/WXF_final_{fn}.csv', index=False)

    # adding the new possible laughters to the list of already ML-detected ones 
    laughters_base = pd.read_json(path_data_standup + f'laughter_detection/fr/{fn}.json').transpose().values
    dfl = LaughtersNormalization(laughters_base, new_laughters)
    os.makedirs('/home/user/data/standup/laughter_detection/test_fr/Updated/', exist_ok=True)
    dfl.to_csv(f'/home/user/data/standup/laughter_detection/test_fr/Updated/{fn}.csv', index=False)