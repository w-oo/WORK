import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpu',
    default = 'cuda:1',
    type = str,
    help='choose gpu device')
parser.add_argument(
    '--data',
    default = 'ICEWS05-15/',
    # default = 'YAGO-WIKI50K/',
    type = str,
    help='choose dataset')
parser.add_argument(
    '--seed',
    default = 5000,
    type = int,
    help='choose the number of align seeds')
parser.add_argument(
    '--dropout',
    default = 0.3,
    type = float,
    help='choose dropout rate')
parser.add_argument(
    '--depth',
    default = 2,
    type = int,
    help='choose number of GNN layers')
parser.add_argument(
    '--gamma',
    default = 3.0,
    type = float,
    help='choose margin')
parser.add_argument(
    '--lr',
    default = 0.005,
    type = float,
    help='choose learning rate')
parser.add_argument(
    '--dim',
    default = 100,
    type = int,
    help='choose embedding dimension')
parser.add_argument( #没有用
    '--eta',
    default = 0.3,
    type = float,
    help='choose Hyper-parameter eta')

parser.add_argument( #没有用
    '--omega',
    default = 0.3,
    type = float,
    help='choose Hyper-parameter omega')
parser.add_argument( #没有用
    '--CF_type',
    # default = "TRO",
    default = "TROR",
    type = str,
    help='choose CF type')
args = parser.parse_args()
