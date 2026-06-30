from itertools import islice
import requests
import rpyc
import os, shutil, logging, time
from subprocess import check_output, call
import heapq, copy, collections

import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdmolops

import config, db

def grep_check(chk_str: str, filename: str) -> bool:
    if os.system(f'grep "{chk_str}" {filename} > /dev/null 2>&1'):
        return False
    else:
        return True

def cmdir(path: str, empty = False) -> None:
    if empty and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok = True)

def connection_test() -> None:
    for node in config.node_info: # node is the node name, str
        try:
            conn = rpyc.connect(config.node_info[node]['host'], config.port)
            print(conn.root.connect_test(node))
            conn.close()
        except:
            print(f'{node} connect failed')

def dict_replace(replace_dict: dict, replace_str: str) -> str:
    for key, value in replace_dict.items():
        replace_str = replace_str.replace(key, str(value))
    return replace_str

def call_job(node: str, job_id: int, job_type: str, nprocs: int, files: dict, db: db.DB, return_list: list[str]):
    conn = rpyc.connect(config.node_info[node]['host'], config.port, keepalive = True, config = {'sync_request_timeout': 86400})
    output = conn.root.run_ORCA(job_id, job_type, list(files.items()), return_list)
    if output:
        storage_dir = os.path.join(config.main_dir, job_type, str(job_id))
        os.makedirs(storage_dir, exist_ok = True)
        for filename, content in output:
            with open(os.path.join(storage_dir, filename), 'wb') as f:
                f.write(content)
    conn.close()
    db.remove_job(job_id)
    db.release_resource(node, nprocs)

def get_stdout(cmd) -> str:
    res = check_output(cmd, shell = True, text = True)
    return res

def pdb2xyz_active(input, output) -> None:
    s = f'cat {input} | awk \'$1 == "ATOM" && $10 == "1.00"\''
    res_l = get_stdout(s).split('\n')
    data = []
    for line in res_l:
        if line:
            data.append(f'{line[76:78]} {line[30:54]}')
    with open(f'{output}', 'w') as f:
        f.write(f'{len(data)}\n\n')
        f.write('\n'.join(data))

def bin_read(filename: str) -> bytes:
    with open(filename, 'rb') as f:
        data = f.read()
    return data

def trash(job_type: str, job_id: int) -> None:
    '''
    Move the job files to trash directory.
    
    Parameters
    ----------
    job_type : str
        The type of the job, either 'opt' or 'neb'.
    job_id : int
        The ID of the job.
    '''
    src_dir = os.path.join(config.main_dir, job_type, str(job_id))
    dst_dir = os.path.join(config.main_dir, config.resource[f'trash_{job_type}_dir'], str(job_id))
    if os.path.exists(src_dir):
        shutil.move(src_dir, dst_dir)

def check_done(logger: logging.Logger, db: db.DB, num_check: int = 3, time_check_f: float = 5.0,  time_check_t: float = 2.0) -> None:
    i = num_check
    while i:
        if db.check_done():
            i -= 1
            time.sleep(time_check_t)
        else:
            i = num_check
            time.sleep(time_check_f)
    logger.info('Job pool empty')

def get_active_region(filename: str) -> np.ndarray:
    '''
    Get the active region from the PDB file.
    
    Parameters
    ----------
    filename : str
        The path to the PDB file.
    
    Returns
    -------
    np.ndarray
        The atom idx of the active region.
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
    active_region_idx = []
    active_region_element = []
    for line in lines:
        if line.startswith('ATOM') and line[62:66] == '1.00':
            active_region_idx.append(int(line[6:11].strip()))
            active_region_element.append(line[76:78].strip())
    return active_region_idx, active_region_element

def parallel_km_rmsd(job_1, job_2, group, mode = 'min-RMSD', res = 'rmsd'):
        coord_1 = job_1.coord
        coord_2 = job_2.coord
        unique_group = np.unique(group) # [0, 1, ..., num_groups-1], fix order
        group_idx = [np.where(group == g)[0] for g in unique_group]
        group_coord_1 = [coord_1[idx] for idx in group_idx]
        group_coord_2 = [coord_2[idx] for idx in group_idx]
        if mode == 'min-RMSD':
            group_dist_matrix = [distance_matrix(g1, g2) ** 2 for g1, g2 in zip(group_coord_1, group_coord_2)]
            _, km_col_idx = zip(*[linear_sum_assignment(dm) for dm in group_dist_matrix])
            group_col_idx = [g_idx[np.argsort(col_idx)] for g_idx, col_idx in zip(group_idx, km_col_idx)]
            res_col_idx= np.zeros_like(group, dtype = int)
            for g_idx, col_idx in zip(group_idx, group_col_idx):
                res_col_idx[g_idx] = col_idx
        elif mode == 'none': # do not make permutation
            res_col_idx = np.arange(coord_1.shape[0])
        elif mode == 'IDPP': # use IDPP module
            # multi_element_group = [(idx, groups) for idx, groups in enumerate(zip(group_coord_1, group_coord_2)) if len(groups[0]) > 1]
            pass
        
        rmsd = np.sqrt(np.mean(np.sum((coord_1 - coord_2[res_col_idx]) ** 2, axis=1)))
        if res == 'rmsd':
            return rmsd
        elif res == 'col_idx':
            return res_col_idx
def _parallel_km_rmsd_wrapper(job_1, job_2, replace_group, mode='min-RMSD', res = 'rmsd'):
    return parallel_km_rmsd(job_1, job_2, replace_group, mode, res)

def parallel_distance(job_1, job_2, group: list, mode, res):
    coord_1 = job_1.coord
    coord_2 = job_2.coord
    n = coord_1.shape[0]
    # group: [eq_type_idx, ] (num_acti_atoms), 0-count
    dis_matrix = cdist(coord_1, coord_2, 'euclidean')
    if mode == 'min-distance':
        cost_matrix = dis_matrix.copy()
        for i in range(n):
            for j in range(i, n):
                if group[i] != group[j]:
                    cost_matrix[i, j] = np.inf
                    cost_matrix[j, i] = np.inf
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
    elif mode == 'min-RMSD':
        cost_matrix = dis_matrix.copy() ** 2
        for i in range(n):
            for j in range(i, n):
                if group[i] != group[j]:
                    cost_matrix[i, j] = np.inf
                    cost_matrix[j, i] = np.inf
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
    elif mode == 'normal':
        row_idx = np.arange(n)
        col_idx = np.arange(n)

    if res == 'rmsd':
        rmsd = np.sqrt(np.mean(np.sum((coord_1[row_idx] - coord_2[col_idx]) ** 2, axis=1)))
        return rmsd
    elif res == 'idx':
        return row_idx, col_idx
    elif res == 'max_cdist':
        return float(max(dis_matrix[row_idx, col_idx]))
    elif res == 'weight_top-3_max_cdist':
        return float(np.sum(np.array([0.1, 0.2, 0.7]) * np.sort(dis_matrix[row_idx, col_idx])[-3:]))
    
def chk_inchikey(xyz_1, xyz_2):
    mol_1 = Chem.MolFromXYZFile(xyz_1)
    mol_2 = Chem.MolFromXYZFile(xyz_2)
    rdDetermineBonds.DetermineConnectivity(mol_1, useVdw = True, covFactor = 1.1)
    rdDetermineBonds.DetermineConnectivity(mol_2, useVdw = True, covFactor = 1.1)
    return Chem.MolToInchiKey(mol_1) == Chem.MolToInchiKey(mol_2)

def clean_ramdisk():
    cmd = 'find ./ramdisk -maxdepth 1 -type d -regextype posix-extended -regex \'.*/[0-9]+$\' -exec rm -r {} +'
    call(f'pdsh -R ssh {",".join([node for node in config.node_info.keys()])} \'{cmd}\'', shell = True)

# def chk_adj_matrix(xyz_1, xyz_2):
#     mol_1 = Chem.MolFromXYZFile(xyz_1)
#     mol_2 = Chem.MolFromXYZFile(xyz_2)
#     rdDetermineBonds.DetermineConnectivity(mol_1, useVdw = True, covFactor = 1.1)
#     rdDetermineBonds.DetermineConnectivity(mol_2, useVdw = True, covFactor = 1.1)
#     return np.array_equal(rdmolops.GetAdjacencyMatrix(mol_1), rdmolops.GetAdjacencyMatrix(mol_2))

def chk_adj_matrix(xyz_1: str, xyz_2: str, eq_group: np.ndarray) -> bool:
    mol_1 = Chem.MolFromXYZFile(xyz_1)
    mol_2 = Chem.MolFromXYZFile(xyz_2)
    rdDetermineBonds.DetermineConnectivity(mol_1, useVdw = True, covFactor = 1.1)
    rdDetermineBonds.DetermineConnectivity(mol_2, useVdw = True, covFactor = 1.1)
    adj_matrix_1 = rdmolops.GetAdjacencyMatrix(mol_1)
    adj_matrix_2 = rdmolops.GetAdjacencyMatrix(mol_2)
    order = np.argsort(eq_group)
    A1 = adj_matrix_1[np.ix_(order, order)]
    A2 = adj_matrix_2[np.ix_(order, order)]
    color = eq_group[order]
    for c in np.unique(color):
        idx = np.where(color == c)[0]
        if not np.array_equal(A1[idx].sum(axis=1), A2[idx].sum(axis=1)):
            return False
    for c in np.unique(color):
        for d in np.unique(color):
            rc = np.where(color == c)[0]
            rd = np.where(color == d)[0]
            M1 = A1[np.ix_(rc, rd)]
            M2 = A2[np.ix_(rc, rd)]
            if not _can_match(M1, M2):
                return False
    return True
def _can_match(M1: np.ndarray, M2: np.ndarray) -> bool:
    if not np.array_equal(np.sort(M1, axis=0), np.sort(M2, axis=0)):
        return False
    if not np.array_equal(np.sort(M1, axis=1), np.sort(M2, axis=1)):
        return False
    return True

def gotify(title: str = None, msg: str = None):
    if config.gotify['enable']:
        resp = requests.post(
            f'http://{config.gotify["host"]}/message?token={config.gotify["token"]}',
            json = {
                'title': title,
                'message': msg,
            }
        )



# ---------- 1. 单对 (s,t) Yen K-shortest ----------
def yen_k_shortest(
        G: nx.DiGraph,
        source,
        target,
        k: int,
        banned,
        weight
):
    """Yen 算法，ban 集为永久不可过节点（虚拟删点）"""
    # 1.1 只保留非 ban 节点
    sub_nodes = [n for n in G.nodes() if n not in banned]
    if source not in sub_nodes or target not in sub_nodes:
        return []
    adj = {n: [] for n in sub_nodes}
    for u, v, d in G.edges(data=True):
        if u in banned or v in banned:
            continue
        w = d.get(weight, 1.0)
        if np.isinf(w):
            continue
        adj[u].append((v, w))
        adj[v].append((u, w))

    # 1.2 Dijkstra 模板（ban 边版本）
    def dijkstra(
            src, dst, ban_edges
    ):
        pq = [(0.0, src, [src])]
        visited = {}
        while pq:
            cost, node, path = heapq.heappop(pq)
            if node in visited:
                continue
            visited[node] = cost
            if node == dst:
                return path, cost
            for nei, w in adj[node]:
                e = (node, nei)
                if e in ban_edges:
                    continue
                if nei not in visited:
                    heapq.heappush(pq, (cost + w, nei, path + [nei]))
        return [], np.inf

    A = []
    B = []

    # 最短
    path_sp, cost_sp = dijkstra(source, target, set())
    if not path_sp:
        return []
    A.append((path_sp, cost_sp))

    for i in range(1, k):
        prev_path, prev_cost = A[-1]
        for j in range(len(prev_path) - 1):
            spur_node = prev_path[j]
            root_path = prev_path[:j + 1]

            # ban root_path 内部边
            ban_e = set()
            for p, q in zip(root_path, root_path[1:]):
                ban_e.add((p, q))
                ban_e.add((q, p))
            # ban 之前最短路径在 root_path 上的相同边
            for p, q in zip(prev_path, prev_path[1:]):
                if p in root_path[:-1] and q in root_path[:-1]:
                    ban_e.add((p, q))
                    ban_e.add((q, p))

            # 临时删点（root_path 除 spur 外）
            temp_ban = set(root_path[:-1])
            temp_adj = {n: [(v, w) for v, w in adj[n] if v not in temp_ban]
                        for n in adj if n not in temp_ban}

            spur_path, spur_cost = dijkstra(spur_node, target, ban_e)
            if spur_path:
                total_path = root_path[:-1] + spur_path
                total_cost = (
                    nx.path_weight(G, root_path, weight) +
                    nx.path_weight(G, list(zip(root_path[-1:], spur_path)), weight)
                )
                heapq.heappush(B, (total_cost, total_path))

        if not B:
            break
        while B:
            c, p = heapq.heappop(B)
            if p not in [pa for pa, _ in A]:
                A.append((p, c))
                break
        else:
            break
    return A


# ---------- 2. 全局 top-k（老算法语义，无损） ----------
def top_k_yen(
        graph: nx.DiGraph,
        r_region,
        p_region,
        k: int = 3,
        safety_factor: int = 5
):
    """
    语义 = top_k_old，但用 Yen + 虚拟删点 + 全局归并。
    safety_factor：每对 (s,t) 先抽 k*safety_factor 条，防止跨对丢失。
    """
    # 2.1 去 inf 边稀疏化
    G = nx.DiGraph()
    for u, v, d in graph.edges(data=True):
        w = d.get("weight", 1.0)
        if not np.isinf(w):
            G.add_edge(u, v, weight=w)

    # 2.2 全局候选池
    global_pool = []
    k_prime = k * safety_factor

    for s in r_region:
        for t in p_region:
            banned = (set(r_region) | set(p_region)) - {s, t}
            for path, cost in yen_k_shortest(G, s, t, k_prime, banned, weight="weight"):
                global_pool.append((cost, path))

    # 2.3 全局取前 k 条（去重）
    global_pool.sort(key=lambda x: x[0])
    out_paths, out_lens = [], []
    seen = set()
    for c, p in global_pool:
        key = tuple(p)
        if key in seen:
            continue
        seen.add(key)
        out_paths.append(p)
        out_lens.append(c)
        if len(out_paths) == k:
            break
    return out_paths, out_lens

def top_k(graph, r_region, p_region, k):
    g = copy.deepcopy(graph)
    paths, lengths = [], []
    done = False
    while (len(paths) < k) and (not done):
        max_steps = []
        p, l = top_k_yen(g, r_region, p_region, len(r_region) * len(p_region))
        if l:
            for path, length in zip(p, l):
                paths.append(path)
                lengths.append(length)
            for path in p:
                step_lengths = []
                for idx in range(len(path) - 1):
                    step = (path[idx], path[idx + 1])
                    step_lengths.append(nx.path_weight(g, step, 'weight'))
                max_step_idx = np.argmax(step_lengths)
                max_steps.append((path[max_step_idx], path[max_step_idx + 1]))
            max_steps = set(max_steps)
            for step in max_steps:
                g[step[0]][step[1]]['weight'] = np.inf
                g[step[1]][step[0]]['weight'] = np.inf
        else:
            done = True
    if len(paths) < k:
        return paths, lengths
    else:
        return paths[:k], lengths[:k]