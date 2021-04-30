import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir')
args = parser.parse_args()

dirname = args.dir
for fn in os.listdir(dirname):
    if os.path.isdir(os.path.join(dirname, fn)):
        continue
    if fn.endswith('.txt'):
        continue
    with open(os.path.join(dirname, fn), 'r') as f_in, open(os.path.join(dirname, fn + '.txt'), 'w') as f_out:
        for line in f_in.readlines():
            parts = line.split('\t')
            f_out.write(parts[1] + '\n')
