#! /usr/bin/env python

launch_script_name = "launch_01_scripts.sh"
launch_script_file = open(launch_script_name, 'w')

block_size = 10
n_blocks = range(20, 200, block_size)
n_steps = 1  

for n_b in n_blocks:

    file_name = 'launch/01_launch_{0}'.format(n_b) + '.pbs'
    launch_script_file.write('qsub ' + file_name + '\n') 
    
    out_file = open(file_name, 'w')
    out_file.write('#!/bin/bash\n')
    out_file.write('\n') 
    out_file.write('#PBS -l nodes=1:ppn=1\n')
    out_file.write('#PBS -l vmem=16gb\n')
    out_file.write('#PBS -l walltime=24:00:00\n')
    out_file.write('#PBS -m ae\n')
    out_file.write('#PBS -M james.ashton.nichols@gmail.com\n')
    out_file.write('\n')
    out_file.write('module remove python/2.7.3')
    out_file.write('module add python/3.5.2')
    out_file.write('\n')
    out_file.write('cd /home/z3180058/projects/pyApproxTools/notebooks/scripts/\n')
    out_file.write('python3 ./01_m_star_tests.py {0} {1} {2}'.format(n_b, n_b+block_size, n_steps) + '\n')
    out_file.close()

