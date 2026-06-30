import pprint, os, multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

import config, utils, db

class NEB():
    '''
    NEB (Nudged Elastic Band) class for performing NEB calculations.
    '''

    def __init__(self, reactant_id: int, product_id: int, idpp_id: int, job_id: int, col_idx: np.ndarray, active_region: np.ndarray) -> None:
        '''
        Initialize the NEB class.

        Parameters
        ----------
        reactant_id : int
            ID of the reactant.
        product_id : int
            ID of the product.
        idpp_id: int
            ID of the IDPP job.
        job_id : int
            ID of the job.
        col_idx : np.ndarray
            Column indices of the product active region, the order is in the active reigon
        active_region : np.ndarray
            Atom indices of the active region.
        '''
        self.reactant_id = reactant_id
        self.product_id = product_id
        self.idpp_id = idpp_id
        self.job_id = job_id
        self.col_idx = col_idx
        self.active_region = np.array(active_region)
        self.have_proc = False
    
    def __repr__(self) -> str:
        '''
        String representation of the NEB class.

        Returns
        -------
        str
            String representation of the NEB class.
        '''
        return pprint.pformat(self.__dict__)

    def prepare_files(self, nimages: int) -> dict:
        '''
        Prepare the NEB calculation input files.

        Returns
        -------
        dict
            Dictionary containing the NEB calculation input files.
            {'filename': b'file content'}
        '''
        nprocs = nimages

        reac_pdb = os.path.join(config.main_dir, 'opt', str(self.reactant_id), 'orca.pdb')
        prod_pdb = os.path.join(config.main_dir, 'opt', str(self.product_id), 'orca.pdb')
        # allxyz = os.path.join(config.main_dir, 'idpp', str(self.idpp_id), 'orca_initial_path.allxyz')
        maxiter = config.neb_info['maxiter']
        orcaff = os.path.join(config.main_dir, config.resource['orcaff'])
        input_template = os.path.join(config.main_dir, 'resource', 'neb.inp')
        replace_dict = {
            'NPROCS': nprocs,
            'NIMAGES': nimages,
            'REAC': f'opt_{self.reactant_id}.pdb',
            'PROD': f'opt_{self.product_id}.pdb',
            'MAXITER': maxiter,
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
            # 'init.allxyz': utils.bin_read(allxyz),
            f'opt_{self.reactant_id}.pdb': utils.bin_read(reac_pdb),
            # f'opt_{self.product_id}.pdb': utils.bin_read(prod_pdb),
            f'opt_{self.product_id}.pdb': ''.join(prod_pdb_content).encode(),
        }
        return output
    
    def run(self, node: str, db: db.DB, nimages: int) -> None:
        '''
        Run the NEB calculation.
        '''
        db.add_job(node, self.job_id)
        # prepare files
        files = self.prepare_files(nimages)
        # call job
        proc = multiprocessing.Process(
            target = utils.call_job,
            args = (node, self.job_id, 'neb', nimages, files, db, config.neb_info['file_need']),
        )
        proc.start()
    
    def process_result(self) -> None:
        '''
        Process the NEB calculation result.
        '''
        job_dir = os.path.join(config.main_dir, 'neb', str(self.job_id))
        orca_log_file = os.path.join(job_dir, 'orca.log')

        self.have_proc = True

        self.normal_exit = True if utils.grep_check('****ORCA TERMINATED NORMALLY****', orca_log_file) else False
        self.converge = True if utils.grep_check('****THE NEB OPTIMIZATION HAS CONVERGED****', orca_log_file) else False
        if self.normal_exit:
            with open(orca_log_file, 'r') as f:
                log_str = f.readlines()
            start_idx = log_str.index('Image Dist.(Ang.)    E(Eh)   dE(kcal/mol)  max(|Fp|)  RMS(Fp)\n') + 1
            end_idx = log_str.index('Straight line distance between images along the path:\n') - 1
            eng, dist = [], []
            for data_line in log_str[start_idx:end_idx]:
                parts = data_line.split()
                eng.append(float(parts[3]))
                dist.append(float(parts[1]))
            eng = np.array(eng)
            dist = np.array(dist)
            extrema_idx = np.sort(np.unique(np.hstack([0, argrelextrema(eng, np.greater_equal)[0], argrelextrema(eng, np.less_equal)[0], len(eng)-1])))
            extrema_value = eng[extrema_idx]
            self.energy = eng
            self.distance = dist
            self.extrema_idx = extrema_idx
            self.extrema_value = extrema_value
            self.minima = argrelextrema(eng, np.less)[0]
            plt.figure()
            plt.plot(eng)
            plt.plot(extrema_idx, extrema_value)
            plt.savefig(os.path.join(job_dir, 'energy_tmp.png'))
            plt.close()
            plt.figure()
            plt.plot(dist, eng)
            plt.savefig(os.path.join(job_dir, 'dist_energy_tmp.png'))
            plt.close()

            utils.cmdir(os.path.join(job_dir, 'images'), empty = True)
            with open(os.path.join(job_dir, 'orca_MEP_trj.xyz'), 'r') as f:
                trj_file = f.readlines()
            num_atoms = float(trj_file[0].strip())
            with open(os.path.join(job_dir, 'orca.pdb'), 'r') as f:
                pdb_file = f.readlines()
            for idx in self.minima:
                xyz_file = trj_file[int(idx * (num_atoms + 2) + 2) : int((idx + 1) * (num_atoms + 2))]
                coord_lines = []
                for line in xyz_file:
                    coord_line = line.strip().split()[1:]
                    line = ''.join([f'{float(coord):>8.3f}' for coord in coord_line])
                    coord_lines.append(line)
                new_file = []
                i = 0
                for line in pdb_file:
                    if line.startswith('ATOM'):
                        new_file.append(line[:30] + coord_lines[i] + line[54:])
                        i += 1
                with open(os.path.join(job_dir, 'images', f'image-{idx}.pdb'), 'w') as f:
                    f.write(''.join(new_file))
            
            self.single_peak()

    def single_peak(self, tol = config.dps_info['NEB_tol']) -> bool:
        '''
        Check if the NEB job has a single peak.

        Parameters
        ----------
        tol : float, optional
            Tolerance for checking single peak, by default config.dps_info['NEB_tol']

        Returns
        -------
        bool
            True if the NEB job has a single peak, False otherwise.
        '''
        values = self.extrema_value
        diff= np.diff(values)
        i, t, r = 0, 0, 0
        while i < len(diff):
            t += diff[i]
            if abs(t) > tol:
                t = 0
                r += 1
            i += 1
        self.single_peak = bool(r == 2)