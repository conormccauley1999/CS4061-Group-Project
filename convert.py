with open('raw_dataset.csv', 'r') as f:
    lns = f.readlines()[1:]
    for ln in lns:
        xs = ln.replace('"', '').split(',')[:-3]
        ys, z = [*map(float, xs[:-1])], int(xs[-1])
        o = []
        for y in ys:
            m = {-1:1,-0.65:2,0.65:3,1:4}
            o.append(m[y])
        o.append(z)
        o = [*map(str, o)]
        print(','.join(o))
