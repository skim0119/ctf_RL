import argparse
import os
import sys
import time

RUNNING_SCRIPT = [
    #"run_multiagent_ppo.py",
    #"run_COMA.py",
    #"run_cvdc.py",
    "run_cvdc2.py",
    #"run_cvdc_comp.py",
    #"run_cvdc3.py", # drawing filtered decoded image
]

parser = argparse.ArgumentParser(description="PPO trainer for convoy")
parser.add_argument("--train_number", type=int, help="training train_number", default=0)
parser.add_argument("--machine", type=str, help="training machine", required=True)
parser.add_argument("--map_size", type=int, help="map size", default=30)
parser.add_argument("--nbg", type=int, help="number of blue ground", default=3)
parser.add_argument("--nba", type=int, help="number of blue air", default=0)
parser.add_argument("--silence", action='store_true', help="call to disable the progress bar")
parser.add_argument("--device", nargs="*", help="GPU numbers")
args = parser.parse_args()

if args.device:
    device = ','.join(args.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = device
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def make_command(run_file_name):
    words = ["python", run_file_name,
             "--train_number", str(args.train_number),
             "--machine", args.machine,
             "--map_size", str(args.map_size),
             "--nbg", str(args.nbg),
             "--nba", str(args.nba),
             "--nrg", str(args.nbg+args.nba),
             "--nra", str(0),
             ]
    command = ' '.join(words)
    if args.silence:
        command += ' --silence'
    command += ' &'
    return command

for script_name in RUNNING_SCRIPT:
    command = make_command(script_name)
    print('Continue with script:')
    print(command)
    os.system(command)
    time.sleep(10)
while True:
    time.sleep(1000)

print(f'Start Time : {time.ctime()}')
print(f'    running: {args.nbg}g{args.nba}a')

