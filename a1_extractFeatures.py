#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import numpy as np
import sys
import argparse
import os
import json
import string
import csv
import re
import pandas as pd
import functools


# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}


def len_dec(fn):  # decorator to automatically take the length of a list returned by 'fn'
    def len_fn(*args, **kwargs):
        return len(fn(*args, **kwargs))
    return len_fn


def flatten(arr):
    # final array only has float values.
    new_arr = []
    for a in arr:
        if isinstance(a, pd.Series):
            new_arr.extend(a.values)
        else:
            new_arr.append(a)
    return new_arr


def extract1(comment):
    ''' This function extracts features from a single comment
      Parameters:
          comment : string, the body of a comment (after preprocessing)
          # bgl: pandas DataFrame of Bristol Gillhooly and Logie features. (to avoid
          #     needing to reopen every time)
          # warr: pandas DataFrame of Warringer features. (to avoid needing to
          #     reopen every time)
      Returns:
          feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    features_array = np.zeros((1, 173))
    body = re.compile("([\w]+|[\W]+)/(?=[\w]+|[\W]+)").findall(comment)  # left
    lemma = re.compile("(?=[\w]+|[\W]+)/([\w]+|[\W]+)").findall(comment)  # right

    # 1. Number of tokens in uppercase ( 3 letters long)
    pattern = re.compile('[A-Z]{3,}\/\S+')
    result = pattern.findall(comment)
    features_array[0][0] = len(result)

    # 2. Number of first-person pronouns
    result = re.compile(r'\b(' + r'|'.join(FIRST_PERSON_PRONOUNS) + r')\b').findall(comment)
    features_array[0][1] = len(result)

    # 3. Number of second-person pronouns
    result = re.compile(r'\b(' + r'|'.join(SECOND_PERSON_PRONOUNS) + r')\b').findall(comment)
    features_array[0][2] = len(result)

    # 4. Number of third-person pronouns
    result = re.compile(r'\b(' + r'|'.join(THIRD_PERSON_PRONOUNS) + r')\b').findall(comment)
    features_array[0][3] = len(result)

    # 5. Number of coordinating conjunctions
    features_array[0][4] = lemma.count('CC')

    # 6. Number of past-tense verbs
    features_array[0][5] = lemma.count('VBD')

    # 7. Number of future-tense verbs
    pattern1 = re.compile('((\'ll\/MD\w*|will\/MD\w*|gonna\/\w+)\s+\w+\/VB)')
    pattern2 = re.compile('(go\/VB\w*\s+to\/TO\w*\s+\w+\/VB)')
    result1 = pattern1.findall(comment)
    result2 = pattern2.findall(comment)
    features_array[0][6] = len(result1) + len(result2)

    # 8. Number of commas
    pattern = re.compile('\S+/,')
    result = pattern.findall(comment)
    features_array[0][7] = len(result)

    # 9. Number of multi-character punctuation tokens
    pattern = re.compile('([?!,;:\.\-`"]{2,})\/')
    result = pattern.findall(comment)
    features_array[0][8] = len(result)

    # 10. Number of common nouns
    features_array[0][9] = lemma.count('NN') + lemma.count('NNS')

    # 11. Number of proper nouns
    features_array[0][10] = lemma.count('NNP') + lemma.count('NNPS')

    # 12. Number of adverbs
    features_array[0][11] = lemma.count('RB') + lemma.count('RBR') + lemma.count('RBS')

    # 13. Number of wh- words
    features_array[0][12] = lemma.count('WDT') + lemma.count('WP') + lemma.count('WP$') + lemma.count('WRB')

    # 14. Number of slang acronyms
    result = re.compile(r'\b(' + r'|'.join(SLANG) + r')\b').findall(comment)
    features_array[0][13] = len(result)

    # prepare 15 - 17
    temp_comment = comment.rstrip('\n')
    sentence_array = temp_comment.split('\n')
    num_tokens = 0
    token_sum = 0
    for sentence in sentence_array:
        pattern = re.compile('\S+\/\S+')
        tokens = pattern.findall(sentence)
        num_tokens += len(tokens)
        for token in tokens:
            token_sum += len(str(token))

    # 15. Average length of sentences, in tokens
    if len(sentence_array) > 0:
        features_array[0][14] = num_tokens / len(sentence_array)

    # 16. Average length of tokens, excluding punctuation-only tokens, in characters
    if num_tokens > 0:
        features_array[0][15] = token_sum / num_tokens

    # 17. Number of sentences.
    features_array[0][16] = len(sentence_array)

    # prepare for 18 - 29
    word_tags = comment.split()
    if len(word_tags) > 0:  # extract just word from each word/tag pair
        retrieve_word = r"(/?\w+)(?=/)"  # they are separated by a / with tag
        extract_words = [findall(retrieve_word, word) for word in word_tags]
        extract_words = [w[0].lower() for w in extract_words if
                         len(w) > 0]

    # 18 - 23
    if len(word_tags) > 0:
        chosen_bgl = []
        for x in extract_words:
            try:
                chosen_bgl.append(BGL_word.loc[x])  # some words might not have a value
            except:
                pass

            # AoA
            AoA = [x.get("AoA (100-700)", np.nan) for x in chosen_bgl]
            AoA = flatten(AoA)

            if np.count_nonzero(~np.isnan(AoA)) > 0:
                # 18. norms average AoA
                features_array[0][17] = np.nanmean(AoA)
                # 21. standard deviation AoA
                features_array[0][20] = np.nanstd(AoA)

            # IMG
            IMG = [x.get("IMG", np.nan) for x in chosen_bgl]
            IMG = flatten(IMG)
            if np.count_nonzero(~np.isnan(IMG)) > 0:
                # 19. average IMG
                features_array[0][18] = np.nanmean(IMG)
                # 22. standard deviation IMG
                features_array[0][21] = np.nanstd(IMG)

            # FAM
            FAM = [x.get("FAM", np.nan) for x in chosen_bgl]
            FAM = flatten(FAM)
            if np.count_nonzero(~np.isnan(FAM)) > 0:
                # 20. average FAM
                features_array[0][19] = np.nanmean(FAM)
                # 23. standard deviation FAM
                features_array[0][22] = np.nanstd(FAM)


    # 24 - 29
    if len(word_tags) > 0:
        chosen_warr = []
        for x in extract_words:
            try:
                chosen_warr.append(warringer_word.loc[x])  # some words might not have a value
            except:
                pass
            #  V.Mean.Sum
            VMS = [x.get("V.Mean.Sum", np.nan) for x in chosen_warr]
            VMS = flatten(VMS)
            if np.count_nonzero(~np.isnan(VMS)) > 0:
                # 24. average V.Mean.Sum
                features_array[0][23] = np.nanmean(VMS)
                # 27. standard deviation V.Mean.Sum
                features_array[0][26] = np.nanstd(VMS)
            #  A.Mean.Sum
            AMS = [x.get("A.Mean.Sum", np.nan) for x in chosen_warr]
            AMS = flatten(AMS)
            if np.count_nonzero(~np.isnan(AMS)) > 0:
                # 25. average A.Mean.Sum
                features_array[0][24] = np.nanmean(AMS)
                # 28. standard deviation A.Mean.Sum
                features_array[0][27] = np.nanstd(AMS)
            # D.Mean.Sum
            DMS = [x.get("D.Mean.Sum", np.nan) for x in chosen_warr]
            DMS = flatten(DMS)
            if np.count_nonzero(~np.isnan(DMS)) > 0:
                # 26. average D.Mean.Sum
                features_array[0][25] = np.nanmean(DMS)
                # 29. standard deviation D.Mean.Sum
                features_array[0][28] = np.nanstd(DMS)

    return features_array


def extract2(feat, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this
        function adds feature 30-173). This should be a modified version of
        the parameter feats.
    '''
    if comment_class == 'Left':
        index = Left_ID.index(comment_id)
        feat[29:-1] = Left_data[index][:]
        feat[-1] = 0

    elif comment_class == 'Center':
        index = Center_ID.index(comment_id)
        feat[29:-1] = Center_data[index][:]
        feat[-1] = 1

    elif comment_class == 'Right':
        index = Right_ID.index(comment_id)
        feat[29:-1] = Right_data[index][:]
        feat[-1] = 2

    elif comment_class == 'Alt':
        index = Alt_ID.index(comment_id)
        feat[29:-1] = Alt_data[index][:]
        feat[-1] = 3

    return feat


def main(args):
    #Declare necessary global variables here.
    path = '/u/cs401/A1/feats/'
    global Alt_data
    Alt_data = np.load(path + 'Alt_feats.dat.npy')
    global Alt_ID
    Alt_ID = open(path + 'Alt_IDs.txt', 'r').read().split('\n')

    global Right_data
    Right_data = np.load(path + 'Right_feats.dat.npy')
    global Right_ID
    Right_ID = open(path + 'Right_IDs.txt', 'r').read().split('\n')

    global Left_data
    Left_data = np.load(path + 'Left_feats.dat.npy')
    global Left_ID
    Left_ID = open(path + 'Left_IDs.txt', 'r').read().split('\n')

    global Center_data
    Center_data = np.load(path + 'Center_feats.dat.npy')  # <class 'numpy.ndarray'>   (200272, 144)
    global Center_ID
    Center_ID = open(path + 'Center_IDs.txt', 'r').read().split('\n')  # List of IDs

    BGL_csv_filename = '/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv'
    Warringer_csv_filename = '/u/cs401/Wordlists/Ratings_Warriner_et_al.csv'

    global BGL
    BGL = pd.read_csv(BGL_csv_filename,
                      usecols=["WORD", "AoA (100-700)", "IMG", "FAM"])
    global warringer
    warringer = pd.read_csv(Warringer_csv_filename,
                            usecols=["Word", "V.Mean.Sum", "D.Mean.Sum", "A.Mean.Sum"])

    global BGL_word
    BGL_word = BGL.set_index(['WORD'])
    global warringer_word
    warringer_word = warringer.set_index(['Word'])

    global findall
    findall= functools.partial(re.findall, flags=re.IGNORECASE)
    global nfindall
    nfindall= len_dec(findall)

    #Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    # Call extract1 for each datatpoint to find the first 29 features.
    # Add these to feats.
    # Call extract2 for each feature vector to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).

    for i in range(feats.shape[0]):
        comment = data[i]['body']
        comment_file = data[i]['cat']
        commentID = data[i]['id']
        feats[i][:-1] = extract1(comment)
        feats[i][:] = extract2(feats[i][:], comment_file, commentID)

        if i % 100 == 0:
            print(i)

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()

    main(args)
