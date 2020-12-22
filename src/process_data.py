# Convert the raw dataset into manageable data

IN_FILE = '..\data\original.csv'
OUT_FILE_STANDARD = '..\data\converted_std.csv'
OUT_FILE_3_CLASS = '..\data\converted_3c.csv'

ANSWER_MAP = {
    -1: 1,
    -0.65: 2,
    0.65: 3,
    1: 4
}

ALIGNMENT_MAP_STANDARD = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7
}

ALIGNMENT_MAP_3_CLASS = {
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 2,
    6: 3,
    7: 3
}


def main(alignment_map, out_file):
    content = ''
    with open(IN_FILE, 'r') as f:
        lines = f.readlines()[1:] # ignore heading line
        for line in lines:
            # get all answers and political alignment
            values = line.replace('"', '').split(',')[:-3]
            # convert answers to floats and alignment to an int
            answers, alignment = [*map(float, values[:-1])], int(values[-1])
            out = []
            for answer in answers:
                out.append(ANSWER_MAP[answer])
            out.append(alignment_map[alignment])
            # convert values to strings
            out = list(map(str, out))
            # join values and add to output
            content += ','.join(out) + '\n'
    with open(out_file, 'w+') as f:
        f.write(content)

main(ALIGNMENT_MAP_STANDARD, OUT_FILE_STANDARD)
main(ALIGNMENT_MAP_3_CLASS, OUT_FILE_3_CLASS)
