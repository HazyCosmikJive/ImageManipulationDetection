import argparse

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')

    parser.add_argument("--resume", default=False, action="store_true", help="If True, resume from last checkpoint.")

    parser.add_argument("--debug", default=False, action="store_true", help="If True, running with debug mode.")

    # distributed
    parser.add_argument('--local_rank', type=int, default=-1, dest='local_rank', help='local rank of current process')
    parser.add_argument('--distributed', action='store_true', dest='distributed', help='Use multi-processing training.')
    parser.add_argument('--gpu', default=[0, 1, 2, 3], nargs='+', type=int, dest='gpu', help='The gpu list used.')

    cfg = parser.parse_args()
    return cfg