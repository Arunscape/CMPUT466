def anneal(a):
    return a / 10

def J(w):
    return w**2


w = 10
a = 1

for i in range(1,100):
    a = anneal(a)
    w = w - a * 2 * w
    print(f"i: {i}  anneal: {a}  w: {w}")
