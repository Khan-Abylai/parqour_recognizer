import numpy


def edit_distance(r, h):
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint8).reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d


def wer(r, h):
    result = 0.0
    for i in range(len(r)):
        if len(r[i]) == 0 and len(h[i]) == 0:
            continue
        if len(r[i]) == 0:
            result += 1.0
            continue
        d = edit_distance(r[i], h[i])
        result += float(d[len(r[i])][len(h[i])]) / len(r[i])

    return result

# res = wer([''],
#           ['sds'])
# print(res)
