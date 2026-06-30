import multiprocessing
import os
import pprint
import numpy as np
import config
import utils
import db

class IDPP():
    '''
    IDPP calss for performing IDPP calculation
    '''
    def __init__(self, reactant_id: int, product_id: int, job_id: int, row_idx: np.ndarray, col_idx: np.ndarray, active_region: np.ndarray) -> None:
        '''
        Initialize the IDPP class.

        Parameters
        ----------
        reactant_id : int
            ID of the reactant.
        product_id : int
            ID of the product.
        job_id : int
            ID of the job.
        col_idx : np.ndarray
            Column indices of the product active region, the order is in the active reigon
        active_region : np.ndarray
            Atom indices of the active region.
        '''
        self.reactant_id = reactant_id
        self.product_id = product_id
        self.job_id = job_id
        assert np.allclose(row_idx, np.arange(row_idx.shape[0]))
        self.col_idx = col_idx
        self.active_region = np.array(active_region)
        self.have_proc = False
    
    def __repr__(self) -> str:
        return pprint.pformat(self.__dict__)
    
    def prepare_files(self) -> dict:
        '''
        Prepare the IDPP input files.

        Returns
        -------
        dict
            Dictionary containing the NEB calculation input files.
            {'filename': b'file content'}
        '''
        nimages = config.neb_info['idpp_nimages']
        reac_pdb = os.path.join(config.main_dir, 'opt', str(self.reactant_id), 'orca.pdb')
        prod_pdb = os.path.join(config.main_dir, 'opt', str(self.product_id), 'orca.pdb')
        orcaff = os.path.join(config.main_dir, config.resource['orcaff'])
        input_template = os.path.join(config.main_dir, 'resource', 'idpp.inp')
        replace_dict = {
            'NIMAGES': nimages,
            'REAC': f'opt_{self.reactant_id}.pdb',
            'PROD': f'opt_{self.product_id}.pdb',
        }
        with open(input_template, 'r') as f:
            input_str = f.read()
        input_str = utils.dict_replace(replace_dict, input_str)
        with open(prod_pdb, 'r') as f:
            prod_pdb_content = f.readlines()
        prod_pdb_active_content = {int(line[6:11]): line for line in prod_pdb_content if line.startswith('ATOM') and int(line[6:11]) in self.active_region}
        for line_idx, line in enumerate(prod_pdb_content):
            if line.startswith('ATOM'):
                atom_idx = int(line[6:11])
                if atom_idx in self.active_region:
                    new_line = line[:30] + f'{prod_pdb_active_content[self.active_region[np.where(self.active_region[self.col_idx] == atom_idx)[0][0]]][30:54]}' + line[54:]
                    prod_pdb_content[line_idx] = new_line
        output = {
            'orca.inp': input_str.encode(),
            'orcaff': utils.bin_read(orcaff),
            f'opt_{self.reactant_id}.pdb': utils.bin_read(reac_pdb),
            # f'opt_{self.product_id}.pdb': utils.bin_read(prod_pdb),
            f'opt_{self.product_id}.pdb': ''.join(prod_pdb_content).encode(),
        }
        return output
    
    def run(self, node: str, db: db.DB) -> None:
        '''
        Run the IDPP calculation.
        '''
        db.add_job(node, self.job_id)
        # prepare files
        files = self.prepare_files()
        # call job
        proc = multiprocessing.Process(
            target = utils.call_job,
            args = (node, self.job_id, 'idpp', 1, files, db, config.neb_info['idpp_file_need']),
        )
        proc.start()
    
    def process_result(self) -> None:
        '''
        Process the IDPP calculation result.
        '''
        job_dir = os.path.join(config.main_dir, 'idpp', str(self.job_id))
        orca_log_file = os.path.join(job_dir, 'orca.log')
        idpp_log_file = os.path.join(job_dir, 'idpp.log')

        self.have_proc = True

        self.normal_exit = True if utils.grep_check('****ORCA TERMINATED NORMALLY****', orca_log_file) else False
        self.converge = True if utils.grep_check('idpp initial path generation successfully converged', orca_log_file) else False
        if self.normal_exit:
            self.barrier = float(utils.get_stdout(f'grep "barrier" {idpp_log_file}').strip().split('\n')[-1].split()[2])
            self.energy = np.array([float(x) for x in utils.get_stdout(f'grep "energy" {idpp_log_file}').strip().split('\n')[-1].split()[2:]])
            self.length = float(utils.get_stdout(f'grep "distance" {idpp_log_file}').strip().split('\n')[-1].split()[-1])
            self.num_valley = int(((self.energy[:-2] > self.energy[1:-1]) & (self.energy[1:-1] < self.energy[2:])).sum())