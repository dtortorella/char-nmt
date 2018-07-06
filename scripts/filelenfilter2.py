import sys

allowed_len = int(sys.argv[1])
source_file = sys.argv[2]
parallel_file = sys.argv[3]

with open(source_file, 'r', encoding='utf-8') as fsrc, open(parallel_file, 'r', encoding='utf-8') as fpar:
    while True:
        line_src = fsrc.readline()
        line_par = fpar.readline()
        if not line_src or not line_par:
            break
        if len(line_src) <= allowed_len:
            print(line_src, end='')
            print(line_par, end='', file=sys.stderr)
