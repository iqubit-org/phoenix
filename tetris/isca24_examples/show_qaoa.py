import pickle
import pdb

def pickle_dump(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def pickle_load(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

tetris_result = pickle_load('tetris_qaoa.pickle')
pdb.set_trace()
with open('my_new_result2.txt', 'r') as file:
    for line in file:
        # Process each line as needed
        name = line.split(',')[0]
        cnot_count = int(line.split(',')[1][1:])
        print(name + ':')
        print('2qan cnot gate count:' + str(cnot_count))
        print('tetris cnot gate count:' + str(tetris_result['myBench/' + name]))