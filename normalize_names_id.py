"""
Script to normalize the names of the downloaded files from yt-dlp with the associated youtube video id.
"""
import pandas as pd
import argparse
import glob
import os
import difflib
from tqdm import tqdm

path_su = '/home/user/data/standup/'

def create_dic(path_csv):

    df = pd.read_csv(path_csv)
    title2id = {}
    for k, v in df[['title', 'url']].values:

        # k = k.replace('|', '｜').replace(':', '：')
        # k = normalize_filename(k)

        title2id[k] = v.split('watch?v=')[-1]

    lang = df.iloc[0].lang if 'lang' in df.columns else ''
    return title2id, lang

def best_match_for_each(list1, list2, force_matching=False, min_ratio=0.8):
    """
    For each string in list2, find and return the string in list1 that
    has the highest similarity ratio according to difflib's SequenceMatcher.
    
    Parameters:
        list1 (list of str): Candidate strings.
        list2 (list of str): Target strings for which to find the best match.
    
    Returns:
        list of str: A list where each element is the best matching string from list1
                     for the corresponding string in list2.
    """
    best_matches = []
    
    for target in list2:
        # For each target string, compute the similarity score with every candidate
        # and choose the candidate with the maximum score.
        best_candidate = max(
            list1,
            key=lambda candidate: difflib.SequenceMatcher(None, candidate, target).ratio()
        )
        best_matches.append(best_candidate)

        if difflib.SequenceMatcher(None, best_candidate, target).ratio() < min_ratio: 
            print("Weird match... best:", best_candidate, '\ntarget:' ,target)
            # if the names really too different
            if not force_matching:
                best_matches.append("")
            else:
                best_matches.append(best_candidate)
        else:
            best_matches.append(best_candidate)

    return best_matches

def normalize_names_id(nameCSV, lang, test):
    """
    Function that can be imported in the descarga script
    """
    title2id, _ = create_dic(path_su+f'CSV_clean/{nameCSV}')
    # the lang used for the files. lang_code is the iso one used for the subtitles
     
    os.path.splitext

    for fn in tqdm(glob.glob(path_su+f'videos/{lang}/*.mp4')):
        # Only process the ones that are not youtube ids
        if ' ' in fn:

            basepath, basename = os.path.split(fn)
            basename, _ = os.path.splitext(basename)

            basename_dic = best_match_for_each(list(title2id.keys()), [basename])[0]
            # print(basename_dic, basename_dic)
            fn_id = os.path.join(basepath, title2id[basename_dic]+'.mp4')
            if not test: 
                os.rename(fn, fn_id)
            else:
                if fn_id == "":
                    print(basename_dic); print(basename, title2id[basename_dic], '\n\n')

            # if there are subtitles
            lang_code = glob.glob(path_su+f'videos/{lang}/*.vtt')[0].split('.')[-2]
            path_vtt = os.path.join(basepath, basename+f'.{lang_code}.vtt')
            path_vtt_id = os.path.join(basepath, title2id[basename_dic]+f'.{lang_code}.vtt')
            if os.path.isfile(path_vtt): 
                if not test: 
                    os.rename(path_vtt, path_vtt_id)
                # else:
                    # print(path_vtt)
                    # print(basename_dic); print(basename, title2id[basename_dic], '\n\n')

    for fn in glob.glob(path_su+f'audio/{lang}/*.wav'):
        # Only process the ones that are not youtube ids
        if ' ' in fn:
            basepath, basename = os.path.split(fn)
            basename, _ = os.path.splitext(basename)
            basename_dic = best_match_for_each(list(title2id.keys()), [basename])[0]
            fn_id = os.path.join(basepath, title2id[basename_dic]+'.wav')
            if not test: 
                os.rename(fn, fn_id)
            # else:
                # print(fn_id)

    for fn in glob.glob(path_su+f'laughter_detection/{lang}/*.json'):
        # Only process the ones that are not youtube ids
        if ' ' in fn:
            basepath, basename = os.path.split(fn)
            basename, _ = os.path.splitext(basename)
            basename_dic = best_match_for_each(list(title2id.keys()), [basename])[0]
            fn_id = os.path.join(basepath, title2id[basename_dic]+'.json')
            if not test: 
                os.rename(fn, fn_id)
            # else:
                # print(fn_id)

def main():
    parser = argparse.ArgumentParser(
        description='Normalize filenames in a folder by removing special characters and converting to lowercase'
    )

    parser.add_argument(
        '-n', '--nameCSV',
        required=True,
        help='Input CSV containing files to rename (they are in /home/user/data/standup/CSV_Clean/ )',
        type=str,
    )

    parser.add_argument(
        '-l', '--lang',
        required=True,
        help='Language (fr, es, es_latam, pt, etc...)',
        type=str,
    )

    parser.add_argument(
        '--test',
        action="store_true",
        required=False,
        help='If just a test',
        # type=bool,
        # default=False,
    )

    args = parser.parse_args()

    normalize_names_id(args.nameCSV, args.lang, args.test)


if __name__ == '__main__':
    main()