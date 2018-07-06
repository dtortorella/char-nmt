import sys

source_file = sys.argv[1]
max_line_length = 0
num_lines = 0

with open(source_file, 'r', encoding='utf-8') as f:
    while True:
        line = f.readline()
        if not line:
            break
        if len(line) > max_line_length:
                max_line_length = len(line)
        num_lines += 1

print(max_line_length, num_lines)
