import sys

allowed_len = int(sys.argv[1])
source_file = sys.argv[2]

with open(source_file, 'r', encoding='utf-8') as f:
    while True:
        line = f.readline()
        if not line:
            break
        if len(line) <= allowed_len:
            print(line, end='')
