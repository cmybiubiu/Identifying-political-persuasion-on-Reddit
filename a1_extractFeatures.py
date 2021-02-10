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

path = '/u/cs401/A1/feats/'
Alt_data = np.load(path + 'Alt_feats.dat.npy')  # <class 'numpy.ndarray'>   (200272, 144)
Alt_ID = open(path + 'Alt_IDs.txt', 'r').read().split('\n')  # List of IDs

Right_data = np.load(path + 'Right_feats.dat.npy')  # <class 'numpy.ndarray'>   (200272, 144)
Right_ID = open(path + 'Right_IDs.txt', 'r').read().split('\n')  # List of IDs

Left_data = np.load(path + 'Left_feats.dat.npy')  # <class 'numpy.ndarray'>   (200272, 144)
Left_ID = open(path + 'Left_IDs.txt', 'r').read().split('\n')  # List of IDs

Center_data = np.load(path + 'Center_feats.dat.npy')  # <class 'numpy.ndarray'>   (200272, 144)
Center_ID = open(path + 'Center_IDs.txt', 'r').read().split('\n')  # List of IDs

BGL_csv_filename = '/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv'
Warringer_csv_filename = '/u/cs401/Wordlists/Ratings_Warriner_et_al.csv'

file = open(BGL_csv_filename, 'r')
reader = csv.reader(file)
word1 = []
AoA = []
IMG = []
FAM = []
for line in reader:
    word1.append(line[1])
    AoA.append(line[3])
    IMG.append(line[4])
    FAM.append(line[5])

file = open(Warringer_csv_filename, "r")
reader = csv.reader(file)
word2 = []
valence = []
arousal = []
dominance = []
for line in reader:
    word2.append(line[1])
    valence.append(line[2])
    arousal.append(line[5])
    dominance.append(line[8])


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
  lemma = re.compile("(?=[\w]+|[\W]+)/([\w]+|[\W]+)").findall(comment) #right

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
  features_array[0][12] = lemma.count('WDT') +  lemma.count('WP') + lemma.count('WP$') + lemma.count('WRB')

  # 14. Number of slang acronyms
  result = re.compile(r'\b(' + r'|'.join(SLANG) + r')\b').findall(comment)
  features_array[0][13] = len(result)

  # 15. Average length of sentences, in tokens
  sentence_amount = comment.count('\n')+1
  features_array[0][14] = len(body)/sentence_amount

  # 16. Average length of tokens, excluding punctuation-only tokens, in characters
  num_token = 0
  sum_token = 0
  for e in body:
      if e[0] not in string.punctuation:
          num_token += 1
          sum_token += len(e)
  if num_token!=0:
    features_array[0][15]

  # 17. Number of sentences.
  features_array[0][16] = comment.count('\n') + 1

  # prepare for 18 - 23
  sAoA = []
  sIMG = []
  sFAM = []
  valid_word_count = 0
  for e in body:
      if e in word1:
          valid_word_count += 1
          i = word1.index(e)
          sAoA.append(int(AoA[i]))
          sIMG.append(int(IMG[i]))
          sFAM.append(int(FAM[i]))

  # 18. Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
  # 19. Average of IMG from Bristol, Gilhooly, and Logie norms
  # 20. Average of FAM from Bristol, Gilhooly, and Logie norms
  # 21. Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms
  # 22. Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
  # 23. Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
  if valid_word_count == 0:
      features_array[0][17:23] = [0,0,0,0,0,0]
  else:
      features_array[0][17] = np.mean(sAoA)
      features_array[0][20] = np.std(sAoA)
      features_array[0][18] = np.mean(sIMG)
      features_array[0][21] = np.std(sIMG)
      features_array[0][19] = np.mean(sFAM)
      features_array[0][22] = np.std(sFAM)

  # prepare for 24 - 29
  s_valence = []
  s_dominance = []
  s_arousal = []
  valid_word_count = 0
  for e in body:
      if e in word2:
          valid_word_count += 1
          i = word2.index(e)
          s_valence.append(float(valence[i]))
          s_dominance.append(float(dominance[i]))
          s_arousal.append(float(arousal[i]))

  # 24. Average of V.Mean.Sum from Warringer norms
  # 25. Average of A.Mean.Sum from Warringer norms
  # 26. Average of D.Mean.Sum from Warringer norms
  # 27. Standard deviation of V.Mean.Sum from Warringer norms
  # 28. Standard deviation of A.Mean.Sum from Warringer norms
  # 29. Standard deviation of D.Mean.Sum from Warringer norms

  if valid_word_count == 0:
      features_array[0][23:29] = [0,0,0,0,0,0]
  else:
      features_array[0][23] = np.mean(s_valence)
      features_array[0][26] = np.std(s_valence)
      features_array[0][24] = np.mean(s_dominance)
      features_array[0][27] = np.std(s_dominance)
      features_array[0][25] = np.mean(s_arousal)
      features_array[0][28] = np.std(s_arousal)

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
        feat[0][29:137] = Left_data[index][:]
        feat[0][137] = 0

    elif comment_class == 'Center':
        index = Center_ID.index(comment_id)
        feat[0][29:137] = Center_data[index][:]
        feat[0][137] = 1

    elif comment_class == 'Right':
        index = Right_ID.index(comment_id)
        feat[0][29:137] = Right_data[index][:]
        feat[0][137] = 2

    elif comment_class == 'Alt':
        index = Alt_ID.index(comment_id)
        feat[0][29:137] = Alt_data[index][:]
        feat[0][137] = 3

    return feat


def main(args):
    #Declare necessary global variables here.

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
        # if data[i]['cat'] == 'Left':
        #     feats[i][-1] = 0
        #     index = Left_ID.index(data[i]['id'])
        #     feats[i][29:-1] = Left_data[index][:]
        #
        # if data[i]['cat'] == 'Center':
        #     feats[i][-1] = 1
        #     index = Center_ID.index(data[i]['id'])
        #     feats[i][29:-1] = Center_data[index][:]
        #
        # if data[i]['cat'] == 'Right':
        #     feats[i][-1] = 2
        #     index = Right_ID.index(data[i]['id'])
        #     feats[i][29:-1] = Right_data[index][:]
        #
        # if data[i]['cat'] == 'Alt':
        #     feats[i][-1] = 3
        #     index = Alt_ID.index(data[i]['id'])
        #     feats[i][29:-1] = Alt_data[index][:]

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
