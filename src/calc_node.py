# Yuhong LIN, linyh8059@gmail.com, Chem. SYSU
# Created: 2025-03-25

# This script is used for calculation node, as a rpyc server, to provide calculation service for the master node.

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer, ForkingServer
import os, shutil
from subprocess import call, check_output, DEVNULL

import config

# check ORCA version
orca_exec = shutil.which('orca')
if config.orca_version not in check_output(f'{orca_exec} -v | grep Version', shell = True, text = True, stderr = DEVNULL):
    raise Exception(f'ORCA {config.orca_version} is not found in PATH')
# check XTB version
if config.xtb_version not in check_output(f'$XTBEXE --version', shell = True, text = True, stderr = DEVNULL):
    raise Exception(f'xTB {config.xtb_version} not match')

localhost = os.uname().nodename

# define the service class
class CalcNodeService(Service):
    def exposed_connect_test(self, node: str) -> str:
        return f'{node} connection success'

    def exposed_run_ORCA(self, job_id: int, job_type: str, files: list[tuple], return_list: list[str]) -> dict:
        '''
        run ORCA job on the node
        input:
            job_id: int, the work_dir is <job_id>
            job_type: str, 'neb' or 'opt'
            files: list, [('filename', b'file content'), ]
        output:
            dict, {'filename': b'file content'}
        '''
        scratch_dir = config.node_info[localhost]['scratch_dir']
        job_dir = os.path.join(scratch_dir, str(job_id))
        os.makedirs(job_dir, exist_ok = True)
        # for filename, content in files.items():
        for filename, content in files:
            with open(os.path.join(job_dir, filename), 'wb') as f:
                f.write(content)
        call(f'{orca_exec} orca.inp > orca.log', shell = True, cwd = job_dir)

        def bread(file):
            with open(file, 'rb') as f:
                res = f.read()
            return res

        if all(x in os.listdir(job_dir) for x in return_list):
            output = [(filename, bread(os.path.join(job_dir, filename))) for filename in return_list]
        elif 'orca.log' in os.listdir(job_dir):
            output = [('orca.log', bread(os.path.join(job_dir, 'orca.log')))]
        else:
            call(f'rm -r {job_dir}', shell = True)
            return False

        # try:
        #     output = [(filename, open(os.path.join(job_dir, filename), 'rb').read()) for filename in return_list]
        # except:
        #     call(f'rm -r {job_dir}', shell = True)
        #     return False

        call(f'rm -r {job_dir}', shell = True)
        return output

if __name__ == '__main__':
    # server = ThreadedServer(CalcNodeService, port = config.port)
    server = ForkingServer(CalcNodeService, port = config.port)
    server.start()