import random
N = 10000
na = nb = 0
b = list(range(3))
for i in range(N):
    # init
    a = [0]*3
    a[random.randint(0, 2)] = 1
    # first choose
    valid_choice = b[:]
    first_index = random.randint(0, 2)
    valid_choice.remove(first_index)
    for j in valid_choice:
        if(a[j] == 1):
            valid_choice.remove(j)
    second_indices = b[:]
    if(len(valid_choice) == 1):
        second_indices.remove(valid_choice[0])
    else:
        second_indices.remove(valid_choice[random.randint(0, 1)])
    if(a[first_index] == 1):
        na += 1
    second_indices.remove(first_index)
    if(a[second_indices[0]] == 1):
        nb += 1
print(na/N, nb/N)