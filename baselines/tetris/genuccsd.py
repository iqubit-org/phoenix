import random
import pickle
from benchmark.mypauli import pauliString

# Function to generate a string of length 'n' filled with 'I's
def generate_string_with_Is(n):
    return 'I' * n

# Function to replace positions in a string with characters from another string
def replace_positions(string, positions, substring):
    new_string = list(string)
    for i, pos in enumerate(positions):
        new_string[pos] = substring[i]
    return ''.join(new_string)

def gen_one_block(n):
    # Generate 8 strings of length 'n' filled with 'I's
    strings = [generate_string_with_Is(n) for _ in range(8)]
    unique_numbers = random.sample(range(n), 4)
    unique_numbers.sort()
    # print(f'unique_numbers: {unique_numbers}')
    # Given string 'ss'
    ss = ['XXYX', 'YXYY', 'XYYY', 'XXXY', 'XYXX', 'YYXY', 'YYYX', 'YXXX']

    # Replace specific positions in each string with characters from 'ss'
    one_block = []
    for i in range(8):
        pauli_string = list(generate_string_with_Is(n))
        for j in range(4):
            # print(f'i is {i}, j is {j},  number is {unique_numbers[j]}')
            pauli_string[unique_numbers[j]] = ss[i][j]
        # insert Z
        for k in range(unique_numbers[0]+1, unique_numbers[1]):
            pauli_string[k] = 'Z'
        for l in range(unique_numbers[2]+1, unique_numbers[3]):
            pauli_string[l] = 'Z'
        one_block.append(pauliString(''.join(pauli_string)))
    return one_block

def generate_n_sqaure_blocks(n):
    blocks = []
    for i in range(n*n):
        blocks.append(gen_one_block(n))
    return blocks

# Change this to your desired length
for n in [10, 15, 20, 25, 30, 35]:
    blocks = generate_n_sqaure_blocks(n)
    with open(f'data/random/random_{12}.pickle', 'wb') as f:
        pickle.dump(blocks, f)
