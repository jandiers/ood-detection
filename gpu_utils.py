# -*- coding: utf-8 -*- 

import os 
import re 
import subprocess
import sys 


def run_command(cmd):
    """Runs terminal commmand and returns output as string.
    
    Args: 
        cmd (str): Terminal command which is to be executed
    
    Returns: 
        string: Terminal output in string format.
    """
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")


def gpu_check():
    """Checks all aviable GPUs for their current workload (gpu_util).
    
    Args: 

    Returns: 
        dict: Dictionary containing gpu id's as keys and gpu workload as 
            values.
    """
    # Run terminal command to retrieve nvidia-smi output 
    output = run_command("nvidia-smi | grep -A1 'GeForce'")
   
    gpu_util_overview = {}
    gpu_id = 0 
    
    # Iterate over terminal output
    for line in output.strip().split("\n")[1::3]:
        # Regex expression captures all integers followed by a % 
        gpu_util = int(re.findall(r"\d+(?:\\.\\d+)?%", line)[1][:-1]) 
        gpu_util_overview[gpu_id] = gpu_util 
        gpu_id += 1 

    return gpu_util_overview 


def pick_gpu(): 
    """ Searches for available GPUs and assigns process.  

    Args: 

    Returns: 
        
    """

    # Check if TensorFlow was already loaded
    assert not ('tensorflow' in sys.modules or 'torch' in sys.modules), \
        'GPU setup must happen before importing TensorFlow/PyTorch'

    gpu_util_overview = gpu_check() 
    for gpu_id, gpu_util in gpu_util_overview.items(): 
        if gpu_util == 0:
            # assign available GPU to process 
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_id)
            return 

    print('Warning, all GPUs are currently in use')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


pick_gpu()
