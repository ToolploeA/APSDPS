import pprint, os, multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

import config, utils, db, glob

class Opt():
    '''
    Opt class for performing optimization calculations.
    '''

    def __init__(self, logger, job_id: int, init: str = None, init_neb_id: int = None, init_image_id: int = None, first: bool = False) -> None:
        '''
        Initialize the Opt class.

        Parameters
        ----------
        job_id : int
            ID of the job.
        init : str, 'reac' or 'prod', optional
            Initial structure for optimization. The default is None.
        init_neb_id : int, optional
            ID of the initial NEB calculation. The default is None.
        init_image_id : int, optional
            ID of the initial image. The default is None.
        first : bool, optional
            Whether this is the first optimization. The default is False.
        '''
        self.job_id = job_id
        self.init = init
        self.init_neb_id = init_neb_id
        self.init_image_id = init_image_id
        self.first = first
        self.logger = logger

    def __repr__(self) -> str:
        '''
        String representation of the Opt class.

        Returns
        -------
        str
            String representation of the Opt class.
        '''
        return pprint.pformat(self.__dict__)
    
    def prepare_files(self) -> dict:
        '''
        Prepare the optimization calculation input files.

        Returns
        -------
        dict
            Dictionary containing the optimization calculation input files.
            {'filename': b'file content'}
        '''
        nprocs = config.opt_info['nprocs']
        orcaff = os.path.join(config.main_dir, config.resource['orcaff'])
        input_template = os.path.join(config.main_dir, 'resource', 'opt.inp')
        if self.first: # job for opt
            if self.init == 'reac':
                if type(config.resource['reactant']) == str:
                    num_struc_reac = 1
                    init_pdb = os.path.join(config.main_dir, config.resource['reactant'])
                elif type(config.resource['reactant']) == list:
                    num_struc_reac = len(config.resource['reactant'])
                    init_pdb = os.path.join(config.main_dir, config.resource['reactant'][self.job_id])
                label = 'reactant.pdb'
            elif self.init == 'prod':
                if type(config.resource['reactant']) == str:
                    num_struc_reac = 1
                elif type(config.resource['reactant']) == list:
                    num_struc_reac = len(config.resource['reactant'])
                if type(config.resource['product']) == str:
                    init_pdb = os.path.join(config.main_dir, config.resource['product'])
                if type(config.resource['product']) == list:
                    init_pdb = os.path.join(config.main_dir, config.resource['product'][self.job_id - num_struc_reac])
                label = 'product.pdb'
        else:
            init_pdb = os.path.join(config.main_dir, 'neb', str(self.init_neb_id), 'images', f'image-{self.init_image_id}.pdb')
            label = f'NEB-{self.init_neb_id}-image-{self.init_image_id}.pdb'
        
        replace_dict = {
            'NPROCS': nprocs,
            'INIT_PDB': label,
        }
        with open(input_template, 'r') as f:
            input_str = f.read()
        input_str = utils.dict_replace(replace_dict, input_str)

        output = {
            'orca.inp': input_str.encode(),
            'orcaff': utils.bin_read(orcaff),
            label: utils.bin_read(init_pdb),
        }
        return output
    
    def run(self, node: str, db: db.DB) -> None:
        '''
        Run the optimization calculation.

        Parameters
        ----------
        node : str
            Node name.
        db : db.DB
            database obj
        '''
        db.add_job(node, self.job_id)
        files = self.prepare_files()
        proc = multiprocessing.Process(
            target = utils.call_job,
            args = (node, self.job_id, 'opt', config.opt_info['nprocs'], files, db, config.opt_info['file_need']),
        )
        proc.start()
    
    def process_result(self) -> None:
        '''
        Process the optimization calculation result.
        '''
        job_dir = os.path.join(config.main_dir, 'opt', str(self.job_id))
        orca_log_file = os.path.join(job_dir, 'orca.log')

        self.normal_exit = True if utils.grep_check('****ORCA TERMINATED NORMALLY****', orca_log_file) else False
        self.converged = True if utils.grep_check('The minimization has converged.', orca_log_file) else False

        if self.converged:
            self.energy = float(utils.get_stdout(f"grep 'FINAL SINGLE POINT ENERGY (QM/MM)' {orca_log_file} | tail -n 1 | awk '{{print $NF}}'"))
            output_pdb = os.path.join(job_dir, 'orca.pdb')
            coord_data = utils.get_stdout(f'cat {output_pdb} | awk \'$1 == "ATOM" && $10 == "1.00"\'').split('\n')
            self.coord = np.array([np.fromstring(s[30:54], sep = ' ', dtype = np.float32) for s in coord_data if s])
            utils.pdb2xyz_active(output_pdb, os.path.join(job_dir, 'orca.xyz'))
