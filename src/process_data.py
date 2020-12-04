# Convert the raw dataset into manageable data

IN_FILE = '..\data\original.csv'
OUT_FILE = '..\data\converted.csv'
ANSWER_MAP = {
    -1: 1,
    -0.65: 2,
    0.65: 3,
    1: 4
}

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
        out.append(alignment)
        # convert values to strings
        out = list(map(str, out))
        # join values and add to output
        content += ','.join(out) + '\n'

with open(OUT_FILE, 'w+') as f:
    f.write(content)
