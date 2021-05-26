import os
import numpy as np
import subprocess
from IPython.core.debugger import set_trace
import sys
import argparse
osp = os.path

def run(instruction, hand_filename, allegro_mapped, graspit):
  filename = osp.join('grasps', '{:s}_grasps.txt'.format(instruction))
  with open(filename, 'r') as f:
    lines = [line.strip() for line in f]
    
  for line in lines:
    object_name, session_num = line.split()
    cmd = 'build/energy_calculator {:s} {:s}_{:s} {:s}'.format(object_name, session_num,
        instruction, hand_filename)
    if allegro_mapped:
      cmd = '{:s} allegro_mapped'.format(cmd)
    else:
      cmd = '{:s} no_allegro_mapped'.format(cmd)
    if graspit:
      cmd = '{:s} graspit'.format(cmd)
    else:
      cmd = '{:s} no_graspit'.format(cmd)

    cmd = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        shell=True, cwd=os.getcwd(), executable='/bin/bash')
    print(cmd.stdout.read())
    print(cmd.stderr.read())
    cmd.wait()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--instruction', required=True)
  parser.add_argument('--hand_filename', required=True)
  parser.add_argument('--allegro_mapped', action='store_true')
  parser.add_argument('--graspit', action='store_true')
  args = parser.parse_args()

  run(args.instruction, args.hand_filename, args.allegro_mapped, args.graspit)
