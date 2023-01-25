import itertools
import numpy as np
import math
import sys
import os
import moving_mnist as mnist
"python create_data.py --dest MovingMNIST --num_digits 1 --motion vertical --num_gifs 1"
"python create_data.py --dest MovingMNIST --csvname test.csv --num_digits 1 --motion vertical horizontal circular_clockwise circular_anticlockwise zigzag tofro  --num_gifs 100"
"python create_data.py --dest MovingMNIST --num_digits 1 --motion circular_clockwise --num_gifs 1"
"python create_data.py --dest MovingMNIST --csvname train.csv --num_digits 1 --motion vertical horizontal circular_clockwise circular_anticlockwise zigzag tofro  --num_gifs 2500"

"python create_data.py --dest MovingMNIST --csvname train_ft.csv --num_digits 1 --motion vertical horizontal circular_clockwise circular_anticlockwise zigzag tofro  --num_gifs 2000"
"python create_data.py --dest MovingMNIST --csvname test_ft.csv --num_digits 1 --motion vertical horizontal circular_clockwise circular_anticlockwise zigzag tofro  --num_gifs 500"

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Command line options')
    # The 'dest' argument is the directory in which to store the generated GIFs
    # The 'num_digits' is the number of digits that move in the GIF
    # The 'motion' argument is the type of motion - 'simple' or 'complex'
    # The 'num_gifs' argument is the no. of GIFs to create
    parser.add_argument('--dest', type=str, dest='dest',
                        default='MovingMNIST')
    parser.add_argument('--csvname', type=str, dest='csvname',
                        default='train.csv')
    parser.add_argument('--num_digits', type=int, dest='num_digits', default=1)
    # parser.add_argument('--motion', type=str, dest='motion', default='simple')
    parser.add_argument('--num_gifs', type=int, dest='num_gifs', default=1)
    parser.add_argument('--motion', nargs='+', required=True)
    args = vars(parser.parse_args(sys.argv[1:]))

    dest = args['dest']
    csvname = args['csvname']
    num_digits = args['num_digits']
    desired_motions = args['motion']
    num_gifs = args['num_gifs']

    numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    allowed_motions = ["vertical", "horizontal", "circular_clockwise",
                       "circular_anticlockwise", "zigzag", "tofro"]

    # Create directory and the captions file
    if not os.path.exists(dest):
        os.makedirs(dest)

    if not os.path.exists(os.path.join(dest, csvname)):
        open(os.path.join(dest, csvname), 'x')

    num_combinations = math.factorial(
        10) // math.factorial(num_digits) // math.factorial(10 - num_digits)

    for i in range(num_gifs):
        digits = list(np.random.randint(low=0, high=10, size=num_digits))
        motions = [desired_motions[np.random.randint(
            len(desired_motions))] for _ in digits]
        mnist.main(digits=digits, motions=motions, dest=dest, csvname=csvname)

   
