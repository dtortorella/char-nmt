import sys

source_file = sys.argv[1]
charset = set('\t')

with open(source_file, 'r', encoding='utf-8') as f:
    while True:
        line = f.readline()
        if line:
            charset.update(line)
        else:
            break

print(''.join(sorted(charset)), end='')
