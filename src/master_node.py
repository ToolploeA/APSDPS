import sys
sys.dont_write_bytecode = True

import rpyc
import pickle, pprint, logging, os, joblib, shutil, glob
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.signal import argrelextrema
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from rdkit import Chem
from pathlib import Path
import network, opt, neb, db, config, utils, rp_region, idpp

# PUBLIC VAR
author = ['Yuhong LIN', ]
email = ['linyh8059@gmail.com / linyh59@mail2.sysu.edu.cn', ]

class Control():
    def __init__(self, logger: logging.Logger) -> None:
        '''
        Initialize the Control class.
        This class is used to control the master node of the AutoEnzyme system.
        '''
        self.neb_jobs = [] # [neb.NEB, ]
        self.opt_jobs = [] # [opt.Opt, ]
        self.idpp_jobs = [] # [odpp.IDPP, ]
        self.graph = network.DPS_Graph()
        self.db = db.DB(os.path.join(config.main_dir, config.resource['db_file']), config.node_info)
        self.logger = logger
        self.opt_todo = [] # [(NEB_id, NEB_image_id), ]
        self.neb_todo = [] # [(reac, prod, idpp_id, n_images)]
        self.idpp_todo = [] # [(reac, prod), ]
        self.idpp_todo_path = [] # [[opt_id1, opt_id2, ], ]
        self.r_region = [] # [node_id, ]
        self.p_region = [] # [node_id, ]
        self.neb_done = dict() # {(reac, prod): neb_id, }
        self.idpp_done = dict() # {(reac, prod): idpp_id, }
        self.zero_path = [] # [{'path': (step, ), 'energy': max_energy, 'max_job': neb_id}, ]
        self.inf_step = set() # {(opt_id1, opt_id2), }
        self.state = 1
        self.active_region, self.element = utils.get_active_region((glob.glob(f'{config.main_dir}/resource/*.pdb') + glob.glob(f'{config.main_dir}/resource/*/*.pdb'))[0]) # np.ndarray, atom idx of the active region, counting starts from 
        self.replace_group = self.generate_replace_group()
        self.num_init_reac = 0
        self.num_init_prod = 0

        os.makedirs(os.path.join(config.main_dir, 'region', 'r'), exist_ok = True)
        os.makedirs(os.path.join(config.main_dir, 'region', 'p'), exist_ok = True)

    def __repr__(self) -> str:
        '''
        String representation of the Control class.

        Returns
        -------
        str
            String representation of the Control class.
        '''
        return pprint.pformat(self.__dict__)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['db']
        return state

    def generate_replace_group(self, rules = 'non C-H'):
        '''
        Generate the replacement group for the active region.
        
        Returns
        -------
        list
            replace group, [eq_type_idx, ]
        '''
        mol = Chem.MolFromPDBFile((glob.glob(f'{config.main_dir}/resource/*.pdb') + glob.glob(f'{config.main_dir}/resource/*/*.pdb'))[0], removeHs = False)
        active_atoms = [mol.GetAtomWithIdx(x - 1) for x in self.active_region]
        if rules == 'non C-H':
            active_H_atoms = [atom for atom in active_atoms if atom.GetSymbol() == 'H']

            # !!! only for this, do NOT push to git server !!!
            # active_H_bonded_atoms = [atom.GetBonds()[0].GetOtherAtom(atom) for atom in active_H_atoms]
            active_H_bonded_atoms = []
            for atom in active_H_atoms:
                if atom.GetBonds():
                    active_H_bonded_atoms.append(atom.GetBonds()[0].GetOtherAtom(atom))
                else:
                    active_H_bonded_atoms.append(mol.GetAtomWithIdx(1))

            eq_group = [[self.active_region.index(atom.GetIdx() + 1) for idx, atom in enumerate(active_H_atoms) if active_H_bonded_atoms[idx].GetSymbol() != 'C']]
        elif rules == 'maunal':
            eq_group_file = config.neb_info['eq_group_file']
            import csv
            with open(eq_group_file, 'r') as f:
                eq_group = [[int(cell) for cell in row] for row in csv.reader(f) if len(row) != 0]
        # eq_group: [[atom_idx in active_atoms of eq_group_1], ([eq_group_2])]
        group_info = [-1] * len(active_atoms)
        group_id = 0
        for group in eq_group:
            for idx in group:
                group_info[idx] = group_id
            group_id += 1
        for idx, point in enumerate(group_info):
            if point == -1:
                group_info[idx] = group_id
                group_id += 1
        return group_info

    def dump(self, file_name: str = os.path.join(config.main_dir, config.resource['control_dump'])) -> None:
        '''
        Dump the current state of the Control class to a file.

        Parameters
        ----------
        file_name : str, optional
            The name of the file to dump the state to, by default 'job.dump'.
        '''
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    def load(self, file_name: str = os.path.join(config.main_dir, config.resource['control_dump'])) -> None:
        '''
        Load the state of the Control class from a file.

        Parameters
        ----------
        file_name : str, optional
            The name of the file to load the state from, by default 'job.dump'.
        '''
        with open(file_name, 'rb') as f:
            control = pickle.load(f)
        self.__dict__.update(control.__dict__)
        self.db = db.DB(os.path.join(config.main_dir, config.resource['db_file']), config.node_info)

    def call_opt(self, init_neb_id: int = None, init_image_id: int = None) -> None:
        '''
        Call the optimization jobs.

        Parameters
        ----------
        num_init_reac: int
            Number of the initial reac
        num_init_prod: int
            Number of the initial prod
        init_neb_id : int, optional
            The ID of the initial NEB job, by default None.
        init_image_id : int, optional
            The ID of the initial image, by default None.
        '''
        job_id = len(self.opt_jobs)
        if job_id < self.num_init_reac:
            init = 'reac'
        elif job_id < (self.num_init_reac + self.num_init_prod):
            init = 'prod'
        else:
            init = None
        job = opt.Opt(self.logger, job_id, init, init_neb_id, init_image_id, first = True if job_id < (self.num_init_reac + self.num_init_prod) else False)
        self.opt_jobs.append(job)
        node = self.db.get_resource(config.opt_info['nprocs'])
        self.logger.info(f'OPT: job {job_id} running on {node}, init: {init}' if init else f'OPT: job {job_id} running on {node}, init: NEB job {init_neb_id} image {init_image_id}')
        job.run(node, self.db)

    def proc_opt_result(self, job: opt.Opt) -> None:
        '''
        Read and process the OPT result
        
        Parameters
        ----------
        job: opt.Opt
            The OPT job
        '''
        job_id = job.job_id
        job.process_result()
        # if job_id == 0:
        #     with open(os.path.join(config.main_dir, 'opt', '0', 'orca.xyz'), 'r') as f:
        #         f.readline()
        #         f.readline()
        #         data = f.readlines()
        
        if job.converged:
            # calc RMSD => if add
            add = True
            # chk C-H topo
            if rp_region.i_region_check(os.path.join(config.main_dir, 'opt', str(job_id), 'orca.xyz'), os.path.join(config.main_dir, 'opt', '0', 'orca.xyz')):
                # rmsd_data = np.array([self.km_rmsd(self.opt_jobs[node], job)[0] for node in self.graph.nodes]) # need parallel
                rmsd_data = np.array(
                    joblib.Parallel(n_jobs = config.dps_info['master_nprocs'])(
                        joblib.delayed(utils.parallel_distance)(
                            self.opt_jobs[node], job, self.replace_group, 'min-RMSD', 'rmsd'
                        ) for node in self.graph.nodes
                    )
                )
                ref_nodes = [list(self.graph.nodes)[int(i)] for i in np.where(rmsd_data < config.dps_info["rmsd_threshold"])[0]]
                if np.any(rmsd_data < config.dps_info['rmsd_threshold_hard']):
                    self.logger.info(f'PROC OPT: job {job_id} not add, RMSD < {config.dps_info["rmsd_threshold_hard"]}')
                    add = False
                elif np.any(rmsd_data < config.dps_info['rmsd_threshold']):
                    same_data = np.array(
                        joblib.Parallel(n_jobs = config.dps_info['master_nprocs'])(
                            joblib.delayed(utils.chk_adj_matrix)(
                                os.path.join(config.main_dir, 'opt', str(job_id), 'orca.xyz'), os.path.join(config.main_dir, 'opt', str(node), 'orca.xyz'), np.array(self.replace_group)
                            ) for node in ref_nodes
                        )
                    )
                    same_nodes = [ref_nodes[int(i)] for i in np.where(same_data)[0]]
                    if np.any(same_data):
                        self.logger.info(f'PROC OPT: job {job_id} not add, RMSD < {config.dps_info["rmsd_threshold"]} and same Adjacency Matrix with {same_nodes}')
                        add = False
            else: # chk C-H topo fail
                self.logger.info(f'PROC OPT: job {job_id} not add, C-H topo check failed')
                add = False

            # add job into graph
            if add:
                # add R, P-region
                # the first R, P
                # if job_id < self.num_init_reac:
                #     self.r_region.append(job_id)
                #     os.symlink(os.path.join(config.main_dir, 'opt', str(job_id), 'orca.xyz'), os.path.join(config.main_dir, 'region', 'r', f'{job_id}.xyz'))
                #     self.logger.info(f'REAC REGION ADD: {job_id}')
                # elif job_id < (self.num_init_reac + self.num_init_prod):
                #     self.p_region.append(job_id)
                #     os.symlink(os.path.join(config.main_dir, 'opt', str(job_id), 'orca.xyz'), os.path.join(config.main_dir, 'region', 'p', f'{job_id}.xyz'))
                #     self.logger.info(f'PROD REGION ADD: {job_id}')
                # else: # normal job
                #     if rp_region.r_region_check(os.path.join(config.main_dir, 'opt', str(job_id), 'orca.xyz')):
                #         # rmsd_data = np.array(
                #         #     joblib.Parallel(n_jobs = config.dps_info['master_nprocs'])(
                #         #         joblib.delayed(utils.parallel_km_rmsd)(
                #         #             self.opt_jobs[node], job, self.replace_group
                #         #         ) for node in self.r_region
                #         #     )
                #         # )
                #         # if np.all(rmsd_data > config.config.dps_info["rmsd_threshold"]):
                #         self.r_region.append(job_id)
                #         os.symlink(os.path.join(config.main_dir, 'opt', str(job_id), 'orca.xyz'), os.path.join(config.main_dir, 'region', 'r', f'{job_id}.xyz'))
                #         self.logger.info(f'REAC REGION ADD: {job_id}')
                #     # R-region and P-region are mutually exclusive
                #     elif rp_region.p_region_check(os.path.join(config.main_dir, 'opt', str(job_id), 'orca.xyz')):
                #         # rmsd_data = np.array(
                #         #     joblib.Parallel(n_jobs = config.dps_info['master_nprocs'])(
                #         #         joblib.delayed(utils.parallel_km_rmsd)(
                #         #             self.opt_jobs[node], job, self.replace_group
                #         #         ) for node in self.p_region
                #         #     )
                #         # )
                #         # if np.all(rmsd_data > config.config.dps_info["rmsd_threshold"]):
                #         self.p_region.append(job_id)
                #         os.symlink(os.path.join(config.main_dir, 'opt', str(job_id), 'orca.xyz'), os.path.join(config.main_dir, 'region', 'p', f'{job_id}.xyz'))
                #         self.logger.info(f'PROD REGION ADD: {job_id}')

                if rp_region.r_region_check(os.path.join(config.main_dir, 'opt', str(job_id), 'orca.xyz')):
                    self.r_region.append(job_id)
                    os.symlink(os.path.join(config.main_dir, 'opt', str(job_id), 'orca.xyz'), os.path.join(config.main_dir, 'region', 'r', f'{job_id}.xyz'))
                    self.logger.info(f'REAC REGION ADD: {job_id}')
                elif rp_region.p_region_check(os.path.join(config.main_dir, 'opt', str(job_id), 'orca.xyz')):
                    self.p_region.append(job_id)
                    os.symlink(os.path.join(config.main_dir, 'opt', str(job_id), 'orca.xyz'), os.path.join(config.main_dir, 'region', 'p', f'{job_id}.xyz'))
                    self.logger.info(f'PROD REGION ADD: {job_id}')

                self.logger.info(f'PROC OPT: job {job_id} add, minimum RMSD: {np.min(rmsd_data)} with node {list(self.graph.nodes)[np.argmin(rmsd_data)]}') if len(rmsd_data) else self.logger.info(f'PROC OPT: job {job_id} add')
                node_list, weight_data = self.edge_weight(job)
                self.graph.new_node(job_id, node_list, weight_data)

            else:
                utils.trash('opt', job_id)
        
        else:
            job.energy = np.inf
            self.logger.info(f'PROC OPT: job {job_id} not add, optimization not converge')
            utils.trash('opt', job_id)

    def proc_all_opt_result(self, all: bool = False) -> None:
        if all:
            self.graph = network.DPS_Graph()
            for job in self.opt_jobs:
                self.proc_opt_result(job)
        else:
            for job in self.opt_jobs:
                if not 'energy' in job.__dict__:
                    self.proc_opt_result(job)
        self.logger.info(f'GRAPH: Number of nodes: {self.graph.number_of_nodes()}')
        # utils.clean_ramdisk()

    def edge_weight(self, job: opt.Opt) -> tuple[list[int], list[float]]:
        '''
        Calculate the weight in the Graph

        Parameters
        ----------
        job: opt.Opt
            The OPT job

        Returns
        -------
        tuple[list[int], list[float]]
            The node list
            The weight list
        '''
        node_list = list(self.graph.nodes)
        # job_id = job.job_id
        # func = lambda job_1, job_2: np.exp(sum([
        #     self.km_rmsd(job_1, job_2)[0],
        #     0.25 * sum([
        #         self.km_rmsd(job_1, self.opt_jobs[0])[0],
        #         self.km_rmsd(job_1, self.opt_jobs[1])[0],
        #         self.km_rmsd(job_2, self.opt_jobs[0])[0],
        #         self.km_rmsd(job_2, self.opt_jobs[1])[0]
        #     ])
        # ]))
        # if job_id in [0, 1]:
        #     weight_data = np.array([np.exp(self.km_rmsd(self.opt_jobs[node], job)[0]) for node in node_list])
        # else:
        #     weight_data = np.array([func(self.opt_jobs[node], job) for node in node_list])

        job_id = job.job_id
        # func = lambda job_1, job_2: sum([
        #     # self.km_rmsd(job_1, job_2)[0],
        #     utils.parallel_km_rmsd(job_1, job_2, self.replace_group),
        #     config.dps_info['edge_weight_ratio'] * 0.25 * sum([
        #         # min([self.km_rmsd(job_1, self.opt_jobs[node])[0] for node in self.r_region]),
        #         # min([self.km_rmsd(job_2, self.opt_jobs[node])[0] for node in self.r_region]),
        #         # min([self.km_rmsd(job_1, self.opt_jobs[node])[0] for node in self.p_region]),
        #         # min([self.km_rmsd(job_2, self.opt_jobs[node])[0] for node in self.p_region]),
        #         min(
        #             joblib.Parallel(n_jobs = config.dps_info['master_nprocs'])(
        #                 joblib.delayed(utils.parallel_km_rmsd)(
        #                     self.opt_jobs[node], job_1, self.replace_group
        #                 )
        #             ) for node in self.r_region
        #         ),
        #         min(
        #             joblib.Parallel(n_jobs = config.dps_info['master_nprocs'])(
        #                 joblib.delayed(utils.parallel_km_rmsd)(
        #                     self.opt_jobs[node], job_2, self.replace_group
        #                 )
        #             ) for node in self.r_region
        #         ),
        #         min(
        #             joblib.Parallel(n_jobs = config.dps_info['master_nprocs'])(
        #                 joblib.delayed(utils.parallel_km_rmsd)(
        #                     self.opt_jobs[node], job_1, self.replace_group
        #                 )
        #             ) for node in self.p_region
        #         ),
        #         min(
        #             joblib.Parallel(n_jobs = config.dps_info['master_nprocs'])(
        #                 joblib.delayed(utils.parallel_km_rmsd)(
        #                     self.opt_jobs[node], job_2, self.replace_group
        #                 )
        #             ) for node in self.p_region
        #         ),
        #     ])
        # ])

        # if job_id in (self.r_region + self.p_region):
        #     # weight_data = np.exp(np.array([self.km_rmsd(self.opt_jobs[node], job)[0] for node in node_list]))
        #     weight_data = np.exp(np.array(
        #         joblib.Parallel(n_jobs = config.dps_info['master_nprocs'])(
        #             joblib.delayed(utils.parallel_km_rmsd)(
        #                 self.opt_jobs[node], job, self.replace_group
        #             ) for node in node_list
        #         )
        #     ))
        # else:
        #     weight_data = np.exp(np.array([func(self.opt_jobs[node], job) for node in node_list]))
        #     # weight_data = np.exp(np.array(
        #     #     joblib.Parallel(n_jobs = config.dps_info['master_nprocs'])(
        #     #         joblib.delayed(func)(
        #     #             self.opt_jobs[node], job
        #     #         )
        #     #     ) for node in node_list
        #     # ))

        # weight_data = np.array([np.exp(self.km_rmsd(self.opt_jobs[node], job)[0]) for node in node_list])
        weight_data = np.array(
            joblib.Parallel(n_jobs = config.dps_info['master_nprocs'])(
                joblib.delayed(utils.parallel_distance)(
                    self.opt_jobs[node], job, self.replace_group, 'min-distance', 'weight_top-3_max_cdist'
                ) for node in node_list
            )
        )
        # weight_data = np.array([utils.parallel_distance(self.opt_jobs[node], job, self.replace_group, 'min-distance', 'weight_top-3_max_cdist') for node in node_list])

        return node_list, weight_data.tolist()

    def rmsd(self, job_1: opt.Opt, job_2: opt.Opt) -> float:
        '''
        Calculate the RMSD of two OPT jobs

        Parameters
        ----------
        job_1: opt.Opt
            The first OPT job
        job_2: opt.Opt
            The second OPT job
        
        Returns
        -------
        float
            The RMSD values
        '''
        return np.sqrt(np.mean(np.sum((job_1.coord - job_2.coord) ** 2, axis = 1)))
    
    # move to utils part, parallel version

    # def km_rmsd(self, job_1: opt.Opt, job_2: opt.Opt, mode = 'min-RMSD') -> tuple[float, np.ndarray]:
    #     '''
    #     Calculate the optimal RMSD of two OPT jobs by Kuhn-Munkres algorithms

    #     Parameters
    #     ----------
    #     job_1: opt.Opt
    #         The first OPT job
    #     job_2: opt.Opt
    #         The second OPT job
        
    #     Returns
    #     -------
    #     float
    #         The RMSD values
    #     np.ndarray
    #         the permutation index act on job_2
    #         job_1.coord <--> job_2.coord[res_col_idx] has the minimum RMSD
    #     '''
    #     group = self.replace_group
    #     coord_1 = job_1.coord
    #     coord_2 = job_2.coord
    #     unique_group = np.unique(group) # [0, 1, ..., num_groups-1], fix order
    #     group_idx = [np.where(group == g)[0] for g in unique_group]
    #     group_coord_1 = [coord_1[idx] for idx in group_idx]
    #     group_coord_2 = [coord_2[idx] for idx in group_idx]
    #     if mode == 'min-RMSD':
    #         group_dist_matrix = [distance_matrix(g1, g2) ** 2 for g1, g2 in zip(group_coord_1, group_coord_2)]
    #     elif mode == 'none': # do not make permutation
    #         group_dist_matrix = [np.fill_diagonal(np.ones(shape = (g1.shape[0], g2.shape[0])), 0) for g1, g2 in zip(group_coord_1, group_coord_2)]
    #     _, km_col_idx = zip(*[linear_sum_assignment(dm) for dm in group_dist_matrix])
    #     group_col_idx = [g_idx[np.argsort(col_idx)] for g_idx, col_idx in zip(group_idx, km_col_idx)]
    #     res_col_idx= np.zeros_like(group, dtype = int)
    #     for g_idx, col_idx in zip(group_idx, group_col_idx):
    #         res_col_idx[g_idx] = col_idx
    #     rmsd = np.sqrt(np.mean(np.sum((coord_1 - coord_2[res_col_idx]) ** 2, axis=1)))
    #     return rmsd, res_col_idx

    def call_neb(self, reactant: int, product: int, idpp_id: int, nimages: int) -> None:
        '''
        Call the NEB jobs.

        Parameters
        ----------
        reactant: int
            the ID of the reactant job
        product: int
            the ID of the product job
        idpp_id: int
            the ID of the corresponding IDPP job
        nimages: int
            the number of n_images, and nprocs
        '''
        job_id = len(self.neb_jobs)
        # col_idx = self.km_rmsd(self.opt_jobs[reactant], self.opt_jobs[product], mode = 'min-RMSD')[1]
        col_idx = self.idpp_jobs[idpp_id].col_idx
        job = neb.NEB(reactant, product, idpp_id, job_id, col_idx, self.active_region)
        self.neb_done[(reactant, product)] = job_id
        self.neb_jobs.append(job)
        node = self.db.get_resource(nimages)
        self.logger.info(f'NEB: job {job_id} running on {node}, reactant id: {reactant}, product id: {product}, n_images: {nimages}')
        job.run(node, self.db, nimages)
    
    def proc_neb_result(self, job: neb.NEB) -> None:
        '''
        Read and process the NEB result
        
        Parameters
        ----------
        job: neb.NEB
            The NEB job
        '''
        job_id = job.job_id
        job.process_result()
        if job.normal_exit:
            condition_zero = [
                len(job.minima) == 0,
                np.max(job.energy) - np.min(job.energy) < 1.0,
                len(argrelextrema(job.energy, np.greater)[0]) == 0,
                len(job.minima) == 2 and np.allclose(job.minima, np.array([1, -1])),
                job.single_peak
            ]
            if any(condition_zero):
                weight = 0
            else:
                weight = np.inf
            self.logger.info(f'PROC NEB: job {job_id} weight: {weight}')
        else:
            weight = np.inf
            self.logger.info(f'PROC NEB: job {job_id} error termination, weight: {weight}')
            utils.trash('neb', job_id)
        self.graph.set_edge_weight(job.reactant_id, job.product_id, weight, job_id)
    
    def proc_all_neb_result(self, all: bool = False) -> None:
        for job in self.neb_jobs:
            if all:
                self.proc_neb_result(job)
            else:
                # if not 'energy' in job.__dict__:
                if not job.have_proc:
                    self.proc_neb_result(job)
        # utils.clean_ramdisk()
    
    def call_idpp(self, reactant: int, product: int) -> None:
        '''
        Call IDPP run

        Parameters
        ----------
        reactant: int
            the ID of the reactant job
        product: int
            the ID of the product job
        '''
        job_id = len(self.idpp_jobs)
        row_idx, col_idx = utils.parallel_distance(self.opt_jobs[reactant], self.opt_jobs[product], self.replace_group, 'min-distance', 'idx')
        job = idpp.IDPP(reactant, product, job_id, row_idx, col_idx, self.active_region)
        self.idpp_done[(reactant, product)] = job_id
        self.idpp_jobs.append(job)
        node = self.db.get_resource(1)
        self.logger.info(f'IDPP: job {job_id} running on {node}, reactant id: {reactant}, product id: {product}')
        job.run(node, self.db)

    def proc_idpp_result(self, job: idpp.IDPP) -> None:
        job.process_result()
        if not job.converge:
            self.idpp_todo_path = list(filter(lambda path: not any([job.reactant_id, job.product_id] == path[i: i+2] for i in range(len(path) - 1)), self.idpp_todo_path))
            self.graph.set_edge_weight(job.reactant_id, job.product_id, weight = np.inf)
            self.logger.info(f'PROC IDPP: IDPP job {job.job_id} not converge, set edge weight to inf')
        else:
            self.logger.info(f'PROC IDPP: IDPP job {job.job_id} succes, barrier = {job.barrier}, length = {job.length}, number of valleys = {job.num_valley}')


    def proc_all_idpp_result(self, all: bool = False) -> None:
        if all:
            for job in self.idpp_jobs:
                # job.process_result()
                self.proc_idpp_result(job)
        else:
            for step in self.idpp_todo:
                job = self.idpp_jobs[self.idpp_done[step]]
                # job.process_result()
                self.proc_idpp_result(job)

    def update_neb_todo(self) -> None:
        '''
        Update NEB TO-DO list
        '''
        self.neb_todo = []
        max_barriers = [] # len = len(self.idpp_todo_path)
        neb_n_images = [] # len = len(self.idpp_todo_path)
        resource_remain = self.db.get_resource_pool_size() * config.neb_info['run_resource_ratio']
        for path in self.idpp_todo_path: # path: [opt_1, opt_2, ...]
            bars = [] # len = len(path) - 1
            n_images = [] # len = len(path) - 1
            for idx in range(len(path)-1):
                step = (path[idx], path[idx + 1])
                idpp_job: idpp.IDPP = self.idpp_jobs[self.idpp_done[step]]
                barrier = idpp_job.barrier
                bars.append(barrier)
                n_images.append(int(np.median([config.neb_info['nimages_min'], config.neb_info['nimages_max'], max(idpp_job.num_valley * config.neb_info['nimages_valleys_ratio'], 2 * int(idpp_job.length))])))
            max_barriers.append(max(bars))
            neb_n_images.append(n_images)
        max_barriers = np.array(max_barriers)
        sorted_idx = np.argsort(max_barriers)
        for idx in sorted_idx:
            path = self.idpp_todo_path[idx] # path: [opt_1, opt_2, ...]
            for i in range(len(path) - 1):
                step = (path[i], path[i + 1])
                if (step not in self.neb_done.keys()) and ((*step, self.idpp_done[step], neb_n_images[idx][i]) not in self.neb_todo) and (resource_remain > neb_n_images[idx][i]): # !!! it will skip the last big one to get the next posible smaller one
                    self.neb_todo.append((*step, self.idpp_done[step], neb_n_images[idx][i]))
                    resource_remain -= neb_n_images[idx][i]
        # if len(self.neb_todo) > config.dps_info['num_neb_explore_step']:
        #     self.neb_todo = self.neb_todo[:config.dps_info['num_neb_explore_step']]

        self.logger.info(f'NEB TO-DO: ({len(self.neb_todo)}) {self.neb_todo}')


    # def update_neb_todo(self) -> None:
    #     '''
    #     Update NEB TO-DO list
    #     '''
    #     self.neb_todo = []
    #     path, length = self.graph.top_k_shortest_path(self.r_region, self.p_region, k = config.dps_info['num_explore_path'], inf_step = self.inf_step)
    #     self.logger.info(f'PATH: The top-{config.dps_info["num_explore_path"]} shortest path: {path}')
    #     self.logger.info(f'PATH: Path length: {[float(x) for x in length]}')
    #     for p in path:
    #         for i in range(len(p) - 1):
    #             step = (p[i], p[i + 1])
    #             if (step not in self.neb_done.keys()) and (step not in self.neb_todo):
    #                 self.neb_todo.append(step)
    #     self.logger.info(f'NEB TO-DO: ({len(self.neb_todo)}) {self.neb_todo}')

    def update_opt_todo(self) -> None:
        '''
        Update Opt TO-DO list
        '''
        self.opt_todo = []
        neb_id = [self.neb_done[step[:2]] for step in self.neb_todo]
        for idx in neb_id:
            neb_job = self.neb_jobs[idx]
            if neb_job.normal_exit:
                self.opt_todo.extend([(idx, int(x)) for x in neb_job.minima])
        self.logger.info(f'OPT TO-DO: ({len(self.opt_todo)}) {self.opt_todo}')

    def update_idpp_todo(self) -> None:
        '''
        Update IDPP TO-DO list
        '''
        self.idpp_todo = []
        path, length = self.graph.top_k_shortest_path(self.r_region, self.p_region, k = config.dps_info['num_idpp_explore_path'], num_zero = len(self.zero_path), inf_step = self.inf_step)
        self.idpp_todo_path = path
        self.logger.info(f'PATH: The top-{config.dps_info["num_idpp_explore_path"]} shortest path: {path}')
        self.logger.info(f'PATH: Path length: {[float(x) for x in length]}')
        for p in path:
            for i in range(len(p) - 1):
                step = (p[i], p[i + 1])
                if (step not in self.idpp_done.keys()) and (step not in self.idpp_todo):
                    self.idpp_todo.append(step)
        self.logger.info(f'IDPP TO-DO: ({len(self.idpp_todo)}) {self.idpp_todo}')

    def new_update_idpp_todo(self) -> None:
        self.idpp_todo = []
        path, length = self.graph.new_top_k_shortest_path(self.r_region, self.p_region, k = config.dps_info['num_idpp_explore_path'], inf_step = self.inf_step)
        self.idpp_todo_path = path
        self.logger.info(f'PATH: The top-{config.dps_info["num_idpp_explore_path"]} shortest path: {path}')
        self.logger.info(f'PATH: Path length: {[float(x) for x in length]}')
        for p in path:
            for i in range(len(p) - 1):
                step = (p[i], p[i + 1])
                if (step not in self.idpp_done.keys()) and (step not in self.idpp_todo):
                    self.idpp_todo.append(step)
        self.logger.info(f'IDPP TO-DO: ({len(self.idpp_todo)}) {self.idpp_todo}')

    def zero_check(self) -> None:
        done = False
        while not done:
            path, length = self.graph.top_k_shortest_path(self.r_region, self.p_region, k = 1, num_zero = len(self.zero_path), inf_step = self.inf_step)
            if (sum(length) == 0) and (len(length) != 0): # new zero-path 
                _inf_step = set()
                for p in path: # p: path in opt, [opt_id, opt_id, ...] length N
                    if p not in [x['path'] for x in self.zero_path]:
                        neb_job_list = [self.neb_done[(p[i], p[i+1])] for i in range(len(p) - 1)] # path in neb, [neb_id, neb_id, ...] length (N - 1)
                        eng_list = [self.neb_jobs[neb_id].energy for neb_id in neb_job_list] # energy of the path, length (N - 1)
                        max_eng = [float(max(x)) for x in eng_list]
                        max_eng_idx = int(np.argmax(max_eng))
                        max_eng_neb_id = neb_job_list[max_eng_idx]
                        max_eng_opt_id = (p[max_eng_idx], p[max_eng_idx+1])

                        self.zero_path.append({
                            'energy': max_eng,
                            'path': p,
                            'neb_job_list': neb_job_list,
                            'max_energy_neb_job': max_eng_neb_id
                        })
                        self.logger.info(f'PATH: New zero-length path: approximate barrier: {max(max_eng)}, energy: {max_eng}, path: {p}, NEB step: {neb_job_list}, maximum energy in step {max_eng_neb_id} ({max_eng_opt_id})')
                        # utils.gotify(title = 'new path found', msg = f'barrier: {max(max_eng)} kcal/mol')
                        _inf_step.add(max_eng_opt_id)

                        path_idx = len(self.zero_path)
                        path_dir = os.path.join(config.main_dir, 'path', str(path_idx))
                        os.makedirs(path_dir)
                        with open(os.path.join(path_dir, 'path.dat'), 'w') as f:
                            f.write('\n'.join(map(str, neb_job_list)))
                        for i, idx in enumerate(neb_job_list):
                            neb_job = self.neb_jobs[idx]
                            os.system(f'cat {os.path.join(config.main_dir, "neb", str(idx), "orca_MEP.activeRegion_trj.xyz")} >> {os.path.join(path_dir, "path-active.xyz")}')
                            if i == 0:
                                path_eng = neb_job.energy
                                path_dist = neb_job.distance
                            else:
                                path_eng = np.hstack((path_eng, neb_job.energy[1:] + path_eng[-1]))
                                path_dist = np.hstack((path_dist, neb_job.distance[1:] + path_dist[-1]))
                        np.savetxt(os.path.join(path_dir, 'energy.dat'), path_eng)
                        np.savetxt(os.path.join(path_dir, 'distance.dat'), path_dist)
                        
                        plt.figure(figsize = (10, 6), dpi = 300)
                        plt.plot(path_dist / path_dist[-1], path_eng, color = 'black')
                        plt.xlabel('Normalized Reaction Coordinate')
                        plt.ylabel('Energy (kcal/mol)')
                        plt.savefig(os.path.join(path_dir, 'eng-dist-draft.png'), bbox_inches = 'tight')
                        plt.close()

                for step in _inf_step:
                    self.inf_step.add(step)
                    self.logger.info(f'GRAPH: modify the highest step edge {step[0]}-{step[1]} to inf weight in path searching')
                    utils.gotify(title = 'new path found', msg = f'barrier: {max(self.neb_jobs[self.neb_done[step]].energy):.2f} kcal/mol')

            else:
                done = True

    def new_zero_check(self) -> None:
        done = False
        while not done:
            path, length = self.graph.new_s_t_shortest_path(self.r_region, self.p_region, len(self.r_region) * len(self.p_region), self.inf_step)
            if np.any(np.isclose(length, 0)): # new zero-path
                _inf_step = set()
                zero_path = [p for idx, p in enumerate(path) if np.isclose(length[idx], 0)]
                for p in zero_path: # p: path in opt, [opt_id, opt_id, ...] length N
                    if p not in [x['path'] for x in self.zero_path]:
                        neb_job_list = [self.neb_done[(p[i], p[i+1])] for i in range(len(p) - 1)] # path in neb, [neb_id, neb_id, ...] length (N - 1)
                        eng_list = [self.neb_jobs[neb_id].energy for neb_id in neb_job_list] # energy of the path, length (N - 1)
                        max_eng = [float(max(x)) for x in eng_list]
                        max_eng_idx = int(np.argmax(max_eng))
                        max_eng_neb_id = neb_job_list[max_eng_idx]
                        max_eng_opt_id = (p[max_eng_idx], p[max_eng_idx+1])

                        self.zero_path.append({
                            'energy': max_eng,
                            'path': p,
                            'neb_job_list': neb_job_list,
                            'max_energy_neb_job': max_eng_neb_id
                        })
                        self.logger.info(f'PATH: New zero-length path: approximate barrier: {max(max_eng)}, energy: {max_eng}, path: {p}, NEB step: {neb_job_list}, maximum energy in step {max_eng_neb_id} ({max_eng_opt_id})')
                        # utils.gotify(title = 'new path found', msg = f'barrier: {max(max_eng)} kcal/mol')
                        _inf_step.add(max_eng_opt_id)

                        path_idx = len(self.zero_path)
                        path_dir = os.path.join(config.main_dir, 'path', str(path_idx))
                        os.makedirs(path_dir)
                        with open(os.path.join(path_dir, 'path.dat'), 'w') as f:
                            f.write('\n'.join(map(str, neb_job_list)))
                        for i, idx in enumerate(neb_job_list):
                            neb_job = self.neb_jobs[idx]
                            os.system(f'cat {os.path.join(config.main_dir, "neb", str(idx), "orca_MEP.activeRegion_trj.xyz")} >> {os.path.join(path_dir, "path-active.xyz")}')
                            if i == 0:
                                path_eng = neb_job.energy
                                path_dist = neb_job.distance
                            else:
                                path_eng = np.hstack((path_eng, neb_job.energy[1:] + path_eng[-1]))
                                path_dist = np.hstack((path_dist, neb_job.distance[1:] + path_dist[-1]))
                        np.savetxt(os.path.join(path_dir, 'energy.dat'), path_eng)
                        np.savetxt(os.path.join(path_dir, 'distance.dat'), path_dist)
                        
                        plt.figure(figsize = (10, 6), dpi = 300)
                        plt.plot(path_dist / path_dist[-1], path_eng, color = 'black')
                        plt.xlabel('Normalized Reaction Coordinate')
                        plt.ylabel('Energy (kcal/mol)')
                        plt.savefig(os.path.join(path_dir, 'eng-dist-draft.png'), bbox_inches = 'tight')
                        plt.close()

                for step in _inf_step:
                    self.inf_step.add(step)
                    self.logger.info(f'GRAPH: modify the highest step edge {step[0]}-{step[1]} to inf weight in path searching')
                    utils.gotify(title = 'new path found', msg = f'barrier: {max(self.neb_jobs[self.neb_done[step]].energy):.2f} kcal/mol')

            else:
                done = True


    def read_path_result(self):
        for path_idx, path in self.zero_path:
            path_dir = os.path.join(config.main_dir, 'path', str(path_idx))
            neb_job_list = path['neb_job_list']
            os.makedirs(path_dir)
            with open(os.path.join(path_dir, 'path.dat'), 'w') as f:
                f.write('\n'.join(map(str, neb_job_list)))
            for idx in range(len(neb_job_list)):
                neb_job = self.neb_jobs[idx]
                if idx == 0:
                    path_eng = neb_job.energy
                    path_dist = neb_job.distance
                else:
                    path_eng = np.hstack((path_eng, neb_job.energy[1:] + path_eng[-1]))
                    path_dist = np.hstack((path_dist, neb_job.distance[1:] + path_dist[-1]))
            np.savetxt(os.path.join(path_dir, 'energy.dat'), path_eng)
            np.savetxt(os.path.join(path_dir, 'distance.dat'), path_dist)
            
            plt.figure(figsize = (10, 6), dpi = 300)
            plt.plot(path_dist / path_dist[-1], path_eng, color = 'black')
            plt.xlabel('Normalized Reaction Coordinate')
            plt.ylabel('Energy (kcal/mol)')
            plt.savefig(os.path.join(path_dir, 'eng-dist-draft.png'), bbox_inches = 'tight')
            plt.close()

if __name__ == '__main__':
    utils.gotify(title = 'Start', msg = 'APS Job Start')

    log_file = os.path.join(config.main_dir, config.resource['log_file'])
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info(f'DPS Reaction Pathway searching scripts')
    for a, e in zip(author, email):
        logger.info(f'Author: {a}')
        logger.info(f'E-mail: {e}')

    ctl = Control(logger)
    try:
        ctl.load()
        logger.info('Restart')        
    except:
        logger.info('New run')

    ctl.new_zero_check()

    # get the number of initial strucs (reac & prod)
    if type(config.resource['reactant']) == str:
        ctl.num_init_reac = 1
    elif type(config.resource['reactant']) == list:
        ctl.num_init_reac = len(config.resource['reactant'])
    if type(config.resource['product']) == str:
        ctl.num_init_prod = 1
    elif type(config.resource['product']) == list:
        ctl.num_init_prod = len(config.resource['product'])
    logger.info(f'Number of initial reactants: {ctl.num_init_reac}, Number of initial products: {ctl.num_init_prod}')

    while not os.path.exists(os.path.join(config.main_dir, 'stop')):
        # 1. Opt

        if len(ctl.neb_jobs) == 0: # init
            for _ in range((ctl.num_init_reac + ctl.num_init_prod)):
                ctl.call_opt()
        else: # not init opt
            for job in ctl.opt_todo:
                ctl.call_opt(*job)

        utils.check_done(logger, ctl.db)

        # ctl.dump()
        ctl.proc_all_opt_result()
        # ctl.dump()

        # 2. read the reac, prod region
        # with open(os.path.join(config.main_dir, config.resource['R_region']), 'r') as f:
        #     ctl.r_region = [int(x.strip()) for x in f.readlines() if x.strip().isdigit()]
        # with open(os.path.join(config.main_dir, config.resource['P_region']), 'r') as f:
        #     ctl.p_region = [int(x.strip()) for x in f.readlines() if x.strip().isdigit()]
        logger.info(f'Reac region: ({len(ctl.r_region)}) {ctl.r_region}')
        logger.info(f'Prod region: ({len(ctl.p_region)}) {ctl.p_region}')

        ctl.new_update_idpp_todo()

        for step in ctl.idpp_todo:
            ctl.call_idpp(*step)
        utils.check_done(logger, ctl.db)

        ctl.proc_all_idpp_result()

        # 3. NEB
        ctl.update_neb_todo()
        # ctl.dump()
        for step in ctl.neb_todo:
            ctl.call_neb(*step)
        
        utils.check_done(logger, ctl.db)

        # ctl.dump()
        ctl.proc_all_neb_result()
        # ctl.dump()

        # 4. check if new zero-length path found
        ctl.dump()
        ctl.new_zero_check()
        # ctl.dump() # one

        ctl.update_opt_todo()

        # 5. check if no opt to-do -> stop
        if (len(ctl.opt_todo) == 0) and (len(ctl.neb_todo) == 0):
            Path(os.path.join(config.main_dir, 'stop')).touch()
    
    else:
        logger.info('user-defined exit')
        logger.info('RESULT:')
        i = 1
        for path in ctl.zero_path:
            logger.info('')
            logger.info(f'ZERO-PATH: {i} ({max(path["energy"])} kcal/mol)')
            i += 1
            for key, value in path.items():
                logger.info(f'ZERO-PATH: {key}: {value}')
        utils.gotify(title = 'DONE', msg = 'DONE')
        # ctl.read_path_result()
        ctl.dump()