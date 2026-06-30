import os

main_dir = '/work/linyh/absgr/042-AutoEnzyme-041_040/work_dir'

resource = {
    # str or list
    'reactant': [f'resource/reactant/{idx}.pdb' for idx in range(48)],
    'product': [f'resource/product/{idx}.pdb' for idx in range(6)],
    'orcaff': 'resource/prmtop.ORCAFF.prms',
    'control_dump': 'job.dump',
    'control_restart': 'job.dump',
    'db_file': 'control.db',
    'log_file': 'main.log',
    'trash_opt_dir': 'trash/opt',
    'trash_neb_dir': 'trash/neb',
}

dps_info = {
    'rmsd_threshold': 0.3,
    'rmsd_threshold_hard': 0.15,
    'num_idpp_explore_path': 100,
    'num_neb_explore_step': 11,
    'n_max': 1,
    'NEB_tol': 1.0,
    'edge_weight_ratio': 0.1,
    'eq_group_file': '.',
    'master_nprocs': 12,
}

neb_info = {
    'idpp_nimages': 32,
    'nimages_min': 16,
    'nimages_max': 32,
    'nimages_valleys_ratio': 6,
    'run_resource_ratio': 0.8,
    'maxiter': 250,
    'file_need': [
        'orca.pdb',
        'orca.log',
        'orca_MEP.activeRegion_trj.xyz',
        'orca_MEP_trj.xyz',
        # 'orca_MEP_ALL.activeRegion_trj.xyz',
        # 'orca_initial_path.activeRegion_trj.xyz',
    ],
    'idpp_file_need': [
        'orca.log',
        # 'orca_initial_path.allxyz',
        'idpp.log',
    ]
}

opt_info = {
    'nprocs': 4,
    'file_need': [
        'orca.pdb',
        'orca.log',
    ]
}

idpp_info = {
    'file_need': [
        'idpp_initial_path.allxyz',
    ]
}

# 30G ramdisk per NEB job
node_info = {
    'hostname':{
        'host': '...',
        'scratch_dir': '...',
        'calc_resource': 44,
        'active': 1,
    },
}

port = 18861

orca_version = '6.1.0'
xtb_version = '6.7.1'

gotify = {
    'enable': True,
    'host': '...',
    'token': '...',
}