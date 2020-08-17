

def IS(array):

    for i in range(1, len(array)):
        key = array[i]
        j = i-1
        while j>=0 and array[j] > key:
            array[j+1] = array[j]
            j -= 1
        array[j+1] = key
    return array



list = [8, 5, 4, 6, 9, 1]

print(IS(list))