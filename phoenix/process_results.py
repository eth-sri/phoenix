# Aggregate from out/*.txt 

for x in ['mnist_mlp2']:

    filename = f'out/{x}.txt'

    #10
    header = ['oABS', 'oC+', 'oC-', 'valid', 'wOK', 'cOK', 'OK', 'RE', 'RE-h', 'DE', 'DE-h', 'P-RE', 'P-RE-h', 'P-DE', 'P-DE-h']

    sums = {}
    for k in header:
        sums[k] = 0

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            vals = line.split(',')[3:]
            vals = vals[:2] + vals[3:7] + vals[8:]
            vals = [int(x) for x in vals]
            assert len(vals) == 14

            sums['oABS'] += 1 - vals[1]
            sums['oC+'] += vals[0] * vals[1]
            sums['oC-'] += (1 - vals[0]) * vals[1]
            for i in range(2, 14):
                sums[header[i+1]] += vals[i]
        for k in sums:
            sums[k] /= len(lines)
            sums[k] *= 100

    print(f'{x}:')
    print(sums)
    print(','.join([str(X) for X in list(sums.keys())]))
    print(','.join([str(X) for X in list(sums.values())]))