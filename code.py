import os
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
import numba as nb
from collections import defaultdict
import re
import shutil
import traceback
import io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# ========================
# 配置参数
# ========================
OUTPUT_MODE = "new"  # "original"或"new"
FORCE_REBUILD = False
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# 输入文件路径
POINT_CSV = r'D:\SJY\Load_Transfer\LoadTransfer_workflow\loadtransfer0920\NODE-0919.csv'
ELEMENT_CSV = r'D:\SJY\Load_Transfer\LoadTransfer_workflow\loadtransfer0920\ELEM-0919.csv'
T_FILE = r'D:\SJY\Load_Transfer\LoadTransfer_workflow\loadtransfer0920\T4.FEM'

# L文件列表
L_FILES = [
    r'D:\SJY\Local\OneWorkFlow\base\oneworflow4projects-globalstrength\concrete_float_cut\Workspace\LoadTransfer\612eabac_006a_L4.FEM',
    # r'D:\SJY\Local\OneWorkFlow\base\oneworflow4projects-globalstrength\concrete_float_cut\Workspace\LoadTransfer\612eabac_006b_L4.FEM',
    # r'D:\SJY\Local\OneWorkFlow\base\oneworflow4projects-globalstrength\concrete_float_cut\Workspace\LoadTransfer\612ecbac_012a_L4.FEM',
    # r'D:\SJY\Local\OneWorkFlow\base\oneworflow4projects-globalstrength\concrete_float_cut\Workspace\LoadTransfer\612ecbac_012b_L4.FEM',
    # r'D:\SJY\Local\OneWorkFlow\base\oneworflow4projects-globalstrength\concrete_float_cut\Workspace\LoadTransfer\612fbaac_005a_L4.FEM',
    # r'D:\SJY\Local\OneWorkFlow\base\oneworflow4projects-globalstrength\concrete_float_cut\Workspace\LoadTransfer\612fbaac_005b_L4.FEM',
    # r'D:\SJY\Local\OneWorkFlow\base\oneworflow4projects-globalstrength\concrete_float_cut\Workspace\LoadTransfer\612fbbac_006_L4.FEM',
    # r'D:\SJY\Local\OneWorkFlow\base\oneworflow4projects-globalstrength\concrete_float_cut\Workspace\LoadTransfer\612gaaac_004a_L4.FEM',
    # r'D:\SJY\Local\OneWorkFlow\base\oneworflow4projects-globalstrength\concrete_float_cut\Workspace\LoadTransfer\612gaaac_004b_L4.FEM',
    # r'D:\SJY\Local\OneWorkFlow\base\oneworflow4projects-globalstrength\concrete_float_cut\Workspace\LoadTransfer\612gaaac_004c_L4.FEM',
    # r'D:\SJY\Local\OneWorkFlow\base\oneworflow4projects-globalstrength\concrete_float_cut\Workspace\LoadTransfer\612gbcac_010_L4.FEM',
    # r'D:\SJY\Local\OneWorkFlow\base\oneworflow4projects-globalstrength\concrete_float_cut\Workspace\LoadTransfer\612gcaac_005_L4.FEM',
]

# 输出目录
OUTPUT_DIR = r'D:\SJY\Load_Transfer\LoadTransfer_workflow\loadtransfer0920\compare'
STRUCTURE_XLSX = os.path.join(OUTPUT_DIR, 'structure.xlsx')
PANEL_XLSX = os.path.join(OUTPUT_DIR, 'panel.xlsx')
MAP_XLSX = os.path.join(OUTPUT_DIR, 'map.xlsx')

# 映射参数
Z_THRESHOLD = 0.0
DISTOL = 10
ANGTOL = 30
MAX_CANDIDATES = 25

# 文件处理参数
TARGET_KEYWORDS = {'BRIGAC', 'BGRAV', 'BEUSLO', 'BELOAD1', 'BELLO2', 'BNLOAD'}
BUFFER_SIZE = 1024 * 1024 * 50  # 50MB
FLUSH_LIMIT = 100000  # 每10万行强制刷新

# ========================
# BlockProcessor类
# ========================
class BlockProcessor:
    def __init__(self):
        self.current_keyword = None
        self.current_condition = None
        self.current_block = []
        self.counters = {
            'total': defaultdict(int),
            'conditions': defaultdict(lambda: defaultdict(int))
        }
        self.condition_data = defaultdict(list)
    
    def process_line(self, line):
        if not line.strip():
            return
        
        first_space = line.find(' ')
        if first_space == -1:
            prefix = line.strip()
        else:
            prefix = line[:first_space].strip()
        
        if prefix in TARGET_KEYWORDS:
            self.finalize_block()
            self._init_new_block(line, prefix, first_space)
        elif self.current_keyword is not None:
            self.current_block.append(line)
    
    def _init_new_block(self, line, prefix, first_space):
        self.current_keyword = prefix
        condition_str = line[first_space+1:].split(None, 1)[0] if first_space != -1 else None
        
        try:
            float(condition_str) if condition_str else None
            self.current_condition = condition_str
            self.current_block = [line]
        except (ValueError, TypeError):
            self.current_condition = None
            self.current_block = []
    
    def finalize_block(self):
        if self.current_condition is None or self.current_keyword is None:
            return
        
        self.counters['total'][self.current_keyword] += 1
        self.counters['conditions'][self.current_condition][self.current_keyword] += 1
        self.condition_data[self.current_condition].extend(self.current_block)
        self.current_keyword = None
        self.current_condition = None
        self.current_block = []
    
    def get_condition_data(self, condition):
        return "".join(self.condition_data.get(condition, []))
    
    def get_all_conditions(self):
        return list(self.condition_data.keys())

# ========================
# PressureMapper类
# ========================
class PressureMapper:
    def __init__(self, mapping_file, struct_file):
        try:
            self.mapping_df = pd.read_excel(mapping_file)
            self.struct_df = pd.read_excel(struct_file)
        except Exception as e:
            print(f"初始化载荷映射器失败: {str(e)}")
            traceback.print_exc()
            raise

    def parse_beuslo(self, data_str):
        """解析BEUSLO数据（从字符串或StringIO对象）"""
        try:
            # 如果是StringIO对象，读取其内容
            if isinstance(data_str, io.StringIO):
                data_str = data_str.getvalue()
            
            pressure_dict = defaultdict(float)
            lines = data_str.splitlines()
            
            i = 0
            while i < len(lines):
                if lines[i].strip().startswith('BEUSLO'):
                    try:
                        case_line = lines[i].split()
                        cell_line = lines[i+1].split()
                        press_line = lines[i+2].split()
                        
                        cell_id = int(float(cell_line[0]))
                        pressures = [float(p) for p in press_line[:4]]
                        avg_pressure = sum(pressures) / 4
                        
                        pressure_dict[(int(float(case_line[1])), cell_id)] += avg_pressure
                        i += 3
                    except (IndexError, ValueError) as e:
                        print(f"解析BEUSLO块失败: {str(e)}")
                        i += 1
                else:
                    i += 1
            
            # 转换为DataFrame
            data = [(cell_id, pressure) for (_, cell_id), pressure in pressure_dict.items()]
            return pd.DataFrame(data, columns=['e_num', 'pressure'])
        except Exception as e:
            print(f"解析BEUSLO数据失败: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def map_pressure(self, panel_df):
        try:
            pressure_dict = panel_df.set_index('e_num')['pressure'].to_dict()
            result_df = self.struct_df[['e_num']].copy()
            result_df = result_df.merge(
                self.mapping_df[['struct_id', 'mapped_panel_id']],
                left_on='e_num',
                right_on='struct_id',
                how='left'
            )
            result_df['new_pressure'] = result_df['mapped_panel_id'].map(pressure_dict)
            return result_df[['e_num', 'new_pressure']]
        except Exception as e:
            print(f"压力映射失败: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()

    def extract_brigac_str(self, data_str):
        data = []
        lines = data_str.splitlines()
        i = 0
        while i < len(lines):
            if lines[i].strip().startswith('BRIGAC'):
                try:
                    load_case = lines[i].split()[1].strip()
                    coord_line = lines[i+1].split()
                    trans_line = lines[i+2].split()
                    rot_line = lines[i+3].split()
                    x, y, z = map(float, coord_line[:3])
                    tx, ty, tz = map(float, trans_line[:3])
                    rx = float(trans_line[3]) if len(trans_line) >= 4 else 0.0
                    ry = float(rot_line[0]) if len(rot_line) >= 1 else 0.0
                    rz = float(rot_line[1]) if len(rot_line) >= 2 else 0.0
                    data.append({
                        '工况': load_case,
                        'X坐标': x, 'Y坐标': y, 'Z坐标': z,
                        'X平移加速度': tx, 'Y平移加速度': ty, 'Z平移加速度': tz,
                        'X旋转加速度': rx, 'Y旋转加速度': ry, 'Z旋转加速度': rz
                    })
                    i += 4
                except Exception as e:
                    i += 1
            else:
                i += 1
        return pd.DataFrame(data)

    def extract_bgrav_str(self, data_str):
        data = []
        lines = data_str.splitlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith('BGRAV'):
                parts = lines[i].split()
                if len(parts) < 2:
                    i += 1
                    continue
                try:
                    case = parts[1].strip()
                    next_line = lines[i+1].split()
                    if len(next_line) < 3:
                        i += 1
                        continue
                    gx, gy, gz = map(float, next_line[:3])
                    data.append([case, gx, gy, gz])
                    i += 2
                except Exception as e:
                    i += 1
            else:
                i += 1
        return pd.DataFrame(data, columns=["工况", "重力X分量", "重力Y分量", "重力Z分量"])

    def process_bnload_str(self, l_data_str, t_file_path):
        def parse_t_file(t_file_path):
            node_coords = {}
            with open(t_file_path, 'r') as file:
                for line in file:
                    if line.startswith('GCOORD'):
                        parts = line.split()
                        try:
                            node = float(parts[1])
                            x = float(parts[2]) * 1000
                            y = float(parts[3]) * 1000
                            z = float(parts[4]) * 1000
                            node_coords[node] = (x, y, z)
                        except (IndexError, ValueError) as e:
                            pass
            return node_coords

        node_coords = parse_t_file(t_file_path)
        data = []
        lines = l_data_str.splitlines()
        i = 0
        while i < len(lines):
            if lines[i].strip().startswith('BNLOAD'):
                try:
                    load_case = lines[i].split()[1]
                    line2 = lines[i+1].split()
                    node = float(line2[0])
                    f1 = float(line2[2]) if len(line2) > 2 else 0.0
                    f2 = float(line2[3]) if len(line2) > 3 else 0.0
                    line3 = lines[i+2].split()
                    f3 = float(line3[0]) if len(line3) > 0 else 0.0
                    f4 = float(line3[1]) if len(line3) > 1 else 0.0
                    f5 = float(line3[2]) if len(line3) > 2 else 0.0
                    f6 = float(line3[3]) if len(line3) > 3 else 0.0
                    
                    if node in node_coords:
                        x, y, z = node_coords[node]
                        data.append({
                            '工况信息': load_case, '节点编号': node,
                            'X坐标': x, 'Y坐标': y, 'Z坐标': z,
                            '自由度1': f1, '自由度2': f2, 
                            '自由度3': f3, '自由度4': f4,
                            '自由度5': f5, '自由度6': f6
                        })
                    i += 3
                except Exception as e:
                    i += 1
            else:
                i += 1
        return pd.DataFrame(data)

    def process_bello2_str(self, l_data_str, t_file_path):
        def parse_t_file(t_file_path):
            gelmnt1, gcoord = {}, {}
            with open(t_file_path, 'r') as f:
                for line in f:
                    if line.startswith('GELMNT1'):
                        try:
                            elem_id = int(float(line.split()[2]))
                            node_line = next(f).split()
                            start = int(float(node_line[0]))
                            end = int(float(node_line[2])) if len(node_line) >= 3 else start
                            gelmnt1[elem_id] = (start, end)
                        except (IndexError, ValueError, StopIteration):
                            pass
                    elif line.startswith('GCOORD'):
                        parts = line.split()
                        try:
                            node_id = int(float(parts[1]))
                            x, y, z = [float(p) * 1000 for p in parts[2:5]]
                            gcoord[node_id] = (x, y, z)
                        except (IndexError, ValueError):
                            pass
            return gelmnt1, gcoord

        gelmnt1, gcoord = parse_t_file(t_file_path)
        data = []
        lines = l_data_str.splitlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith('BELLO2'):
                try:
                    case = float(lines[i].split()[1])
                    elem_line = lines[i+1].split()
                    elem_id = int(float(elem_line[0]))
                    load_line = lines[i+2].split()
                    l1 = float(load_line[1]) if len(load_line) > 1 else 0.0
                    l2 = float(load_line[2]) if len(load_line) > 2 else 0.0
                    l3 = float(load_line[3]) if len(load_line) > 3 else 0.0
                    
                    start, end = gelmnt1.get(elem_id, (None, None))
                    start_coord = gcoord.get(start, (None, None, None))
                    end_coord = gcoord.get(end, (None, None, None))
                    
                    data.append({
                        '工况': case, '单元编号': elem_id,
                        '起点X': start_coord[0], '起点Y': start_coord[1], '起点Z': start_coord[2],
                        '终点X': end_coord[0], '终点Y': end_coord[1], '终点Z': end_coord[2],
                        '自由度1载荷': l1, '自由度2载荷': l2, '自由度3载荷': l3
                    })
                    i += 3
                except Exception as e:
                    i += 1
            else:
                i += 1
        return pd.DataFrame(data)

# ========================
# 预处理函数
# ========================
def build_coord_dict(point_csv):
    try:
        file_size = os.path.getsize(point_csv)
        chunk_size = 100000
        coord_dict = {}
        
        with tqdm(total=file_size, unit='B', unit_scale=True, desc='Building coordinate dictionary') as pbar:
            for chunk in pd.read_csv(point_csv, usecols=['NODE', 'X', 'Y', 'Z'], 
                                   dtype={'NODE': int}, chunksize=chunk_size, encoding='utf-8'):
                chunk = chunk.set_index('NODE') * 1000
                chunk_dict = chunk.apply(lambda x: (x.X, x.Y, x.Z), axis=1).to_dict()
                coord_dict.update(chunk_dict)
                pbar.update(chunk.memory_usage(index=True, deep=True).sum())
        
        return coord_dict
    except Exception as e:
        print(f"构建坐标字典失败: {str(e)}")
        traceback.print_exc()
        return {}

def process_elements(element_csv, coord_dict, output_xlsx):
    try:
        def count_rows():
            with open(element_csv, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f) - 1
        
        total_rows = count_rows()
        reader = pd.read_csv(
            element_csv,
            chunksize=100000,
            usecols=['e_num', 'point1', 'point2', 'point3', 'point4'],
            dtype={'e_num': int, 'point1': 'Int64', 'point2': 'Int64',
                   'point3': 'Int64', 'point4': 'Int64'},
            encoding='utf-8'
        )
        
        with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
            for chunk_idx, chunk in enumerate(reader):
                data = []
                for _, row in tqdm(chunk.iterrows(), total=len(chunk), 
                                 desc=f'Processing chunk {chunk_idx+1}'):
                    record = {'e_num': row['e_num']}
                    for i, p in enumerate(['point1', 'point2', 'point3', 'point4'], 1):
                        node = row[p]
                        if pd.isna(node):
                            for axis in ['x', 'y', 'z']:
                                record[f"{axis}{i}"] = None
                        else:
                            coord = coord_dict.get(int(node), (None, None, None))
                            record[f"x{i}"] = coord[0]
                            record[f"y{i}"] = coord[1]
                            record[f"z{i}"] = coord[2]
                    data.append(record)
                
                df = pd.DataFrame(data)
                df.to_excel(writer, index=False, sheet_name=f'Chunk_{chunk_idx+1}')
        
        return output_xlsx
    except Exception as e:
        print(f"处理单元数据失败: {str(e)}")
        traceback.print_exc()
        return None

# ========================
# 步骤函数
# ========================
def step1_preprocess():
    try:
        print("步骤1: 预处理东大模型...")
        start_time = time.time()
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        if os.path.exists(STRUCTURE_XLSX) and not FORCE_REBUILD:
            print(f"检测到结构数据文件已存在: {STRUCTURE_XLSX}")
            print("跳过步骤1计算")
            return True
        
        coord_dict = build_coord_dict(POINT_CSV)
        if not coord_dict:
            raise Exception("坐标字典为空，无法继续处理")
        
        process_elements(ELEMENT_CSV, coord_dict, STRUCTURE_XLSX)
        
        end_time = time.time()
        print(f"步骤1完成，耗时: {end_time - start_time:.2f}秒")
        return True
    except Exception as e:
        print(f"步骤1失败: {str(e)}")
        traceback.print_exc()
        return False

def extract_first_condition(input_file, output_dir):
    try:
        print(f"步骤2: 从L文件中提取第一个完整工况...")
        start_time = time.time()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用全局常量
        global TARGET_KEYWORDS, BUFFER_SIZE
        
        blocks = []
        first_condition = None
        current_block = []
        collecting_block = False
        condition_value = None
        
        with open(input_file, 'r', buffering=BUFFER_SIZE) as f:
            for line in f:
                first_space = line.find(' ')
                if first_space == -1:
                    prefix = line.strip()
                else:
                    prefix = line[:first_space].strip()
                    
                if prefix in TARGET_KEYWORDS:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            condition_str = parts[1]
                            current_condition = float(condition_str)
                            
                            if first_condition is None:
                                first_condition = current_condition
                                condition_value = condition_str
                                print(f"发现第一个工况: {condition_str}")
                            
                            if collecting_block:
                                blocks.append(current_block.copy())
                                current_block = []
                            collecting_block = True
                            current_block = [line]
                        except (ValueError, IndexError):
                            collecting_block = False
                            current_block = []
                    else:
                        collecting_block = False
                        current_block = []
                elif collecting_block:
                    current_block.append(line)
        
        if collecting_block and current_block:
            blocks.append(current_block)
        
        if first_condition:
            first_condition_blocks = [block for block in blocks if block[0].split()[1] == condition_value]
            
            if first_condition_blocks:
                output_path = os.path.join(output_dir, f"{condition_value}.txt")
                with open(output_path, 'w') as out_f:
                    for block in first_condition_blocks:
                        for line in block:
                            out_f.write(line)
                
                end_time = time.time()
                print(f"已保存第一个工况文件: {output_path}, 耗时: {end_time - start_time:.2f}秒")
                return output_path
        
        return None
    except Exception as e:
        print(f"提取第一个工况失败: {str(e)}")
        traceback.print_exc()
        return None

def step2_extract_first_condition(l_file):
    try:
        temp_dir = os.path.join(OUTPUT_DIR, "temp_first_condition")
        os.makedirs(temp_dir, exist_ok=True)
        
        file_id = os.path.basename(l_file).split('.')[0]
        condition_file = os.path.join(temp_dir, f"{file_id}_first_condition.txt")
        
        if os.path.exists(condition_file) and not FORCE_REBUILD:
            print(f"检测到已有第一个工况文件: {condition_file}")
            return condition_file
        
        condition_file = extract_first_condition(l_file, temp_dir)
        return condition_file
    except Exception as e:
        print(f"步骤2失败: {str(e)}")
        traceback.print_exc()
        return None

def step3_extract_pressure(condition_file, t_file):
    try:
        print("步骤3: 提取面压力数据...")
        start_time = time.time()
        
        if os.path.exists(PANEL_XLSX) and not FORCE_REBUILD:
            print(f"检测到面板数据文件已存在: {PANEL_XLSX}")
            print("跳过步骤3计算")
            return PANEL_XLSX
        
        def parse_beuslo(filename):
            pressure_dict = {}
            with open(filename, 'r') as f:
                lines = f.readlines()
            i = 0
            while i < len(lines):
                if lines[i].strip().startswith('BEUSLO'):
                    try:
                        case_line = lines[i].split()
                        cell_line = lines[i+1].split()
                        press_line = lines[i+2].split()
                        cell_id = int(float(cell_line[0]))
                        pressures = list(map(float, press_line[:4]))
                        avg_pressure = sum(pressures) / 4
                        pressure_dict[cell_id] = avg_pressure
                        i += 3
                    except (IndexError, ValueError):
                        i += 1
                else:
                    i += 1
            return pressure_dict

        def parse_gelmt(filename, target_cells):
            gelmt_dict = {}
            required_nodes = set()
            with open(filename, 'r') as f:
                while True:
                    line1 = f.readline()
                    if not line1: 
                        break
                    if line1.startswith('GELMNT1'):
                        try:
                            cell_id = int(float(line1.split()[2]))
                            if cell_id not in target_cells:
                                f.readline()
                                f.readline()
                                continue
                            line2 = f.readline().split()
                            line3 = f.readline().split()
                            if len(line2) == 4 and len(line3) == 4:
                                nodes = [
                                    int(float(line2[0])), 
                                    int(float(line2[2])),
                                    int(float(line3[0])),
                                    int(float(line3[2]))
                                ]
                            elif len(line2) == 4 and len(line3) == 2:
                                nodes = [
                                    int(float(line2[0])), 
                                    int(float(line2[1])),
                                    int(float(line2[2]))
                                ]
                            else:
                                continue
                            gelmt_dict[cell_id] = nodes
                            required_nodes.update(nodes)
                        except (IndexError, ValueError):
                            continue
            return gelmt_dict, required_nodes

        def parse_gcoord(filename, required_nodes):
            gcoord_dict = {}
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('GCOORD'):
                        parts = line.strip().split()
                        try:
                            node_id = int(float(parts[1]))
                            if node_id in required_nodes:
                                coords = [float(x)*1000 for x in parts[2:5]]
                                gcoord_dict[node_id] = coords
                        except:
                            pass
            return gcoord_dict

        def build_final_data(beuslo_data, gelmt_dict, gcoord_dict):
            final_data = []
            for cell_id, avg_pressure in beuslo_data.items():
                node_ids = gelmt_dict.get(cell_id, [])
                row = {'e_num': cell_id}
                for i in range(1, 5):
                    if i <= len(node_ids):
                        coords = gcoord_dict.get(node_ids[i-1], [None]*3)
                        row[f'x{i}'] = coords[0]
                        row[f'y{i}'] = coords[1]
                        row[f'z{i}'] = coords[2]
                    else:
                        row[f'x{i}'] = None
                        row[f'y{i}'] = None
                        row[f'z{i}'] = None
                row['pressure'] = avg_pressure
                final_data.append(row)
            return final_data

        beuslo_data = parse_beuslo(condition_file)
        target_cells = set(beuslo_data.keys())
        gelmt_dict, required_nodes = parse_gelmt(t_file, target_cells)
        gcoord_dict = parse_gcoord(t_file, required_nodes)
        final_data = build_final_data(beuslo_data, gelmt_dict, gcoord_dict)
        
        df = pd.DataFrame(final_data)
        df.to_excel(PANEL_XLSX, index=False, float_format='%.5f')
        
        end_time = time.time()
        print(f"步骤3完成，耗时: {end_time - start_time:.2f}秒")
        return PANEL_XLSX
    except Exception as e:
        print(f"步骤3失败: {str(e)}")
        traceback.print_exc()
        return None

@nb.njit(cache=True)
def panel_normal(vertices):
    if vertices.shape[0] == 3:
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
    else:
        v1 = vertices[2] - vertices[0]
        v2 = vertices[3] - vertices[1]
    cross = np.empty(3, dtype=np.float64)
    cross[0] = v1[1]*v2[2] - v1[2]*v2[1]
    cross[1] = v1[2]*v2[0] - v1[0]*v2[2]
    cross[2] = v1[0]*v2[1] - v1[1]*v2[0]
    norm = np.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
    return cross / norm if norm > 1e-6 else np.zeros(3)

@nb.njit(cache=True)
def panel_area(vertices):
    if vertices.shape[0] == 3:
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
    else:
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
    cross = np.array([
        v1[1]*v2[2] - v1[2]*v2[1],
        v1[2]*v2[0] - v1[0]*v2[2],
        v1[0]*v2[1] - v1[1]*v2[0]
    ])
    area = 0.5 * np.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
    if vertices.shape[0] == 4:
        v3 = vertices[3] - vertices[0]
        cross2 = np.array([
            v2[1]*v3[2] - v2[2]*v3[1],
            v2[2]*v3[0] - v2[0]*v3[2],
            v2[0]*v3[1] - v2[1]*v3[0]
        ])
        area += 0.5 * np.sqrt(cross2[0]**2 + cross2[1]**2 + cross2[2]**2)
    return area

@nb.njit(cache=True)
def compute_shadow_area(centroid, edges):
    """计算投影面积"""
    total = 0.0
    for i in range(edges.shape[0]):
        p1 = edges[i, 0]
        p2 = edges[i, 1]
        v1 = p1 - centroid
        v2 = p2 - centroid
        cross = np.array([
            v1[1]*v2[2] - v1[2]*v2[1],
            v1[2]*v2[0] - v1[0]*v2[2],
            v1[0]*v2[1] - v1[1]*v2[0]
        ])
        total += 0.5 * np.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
    return total

def load_data(file_path):
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        parquet_file = os.path.join(OUTPUT_DIR, f"{base_name}.parquet")
        
        if not FORCE_REBUILD and os.path.exists(parquet_file):
            df = pd.read_parquet(parquet_file)
            df['vertices'] = df['vertices'].apply(lambda lst: np.array(lst).reshape(-1, 3))
            df['centroid'] = df['centroid'].apply(lambda c: np.array(c).flatten())
            df['normal'] = df['normal'].apply(lambda n: np.array(n).flatten())
            return df
        
        print(f"预处理 [{os.path.basename(file_path)}]...")
        df = pd.read_excel(file_path, sheet_name=0)
        vertices_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="解析顶点"):
            coords = []
            for i in range(1, 5):
                if f'x{i}' in row and not pd.isna(row[f'x{i}']):
                    coords.append([row[f'x{i}'], row[f'y{i}'], row[f'z{i}']])
            if coords:
                vertices_list.append(np.array(coords))
        
        df = df.iloc[:len(vertices_list)].copy()
        df['vertices'] = vertices_list
        df['centroid'] = [vs.mean(axis=0) for vs in vertices_list]
        df['normal'] = [panel_normal(vs) for vs in vertices_list]
        df['area'] = [panel_area(vs) for vs in vertices_list]
        
        df_cache = df.copy()
        df_cache['vertices'] = df_cache['vertices'].apply(lambda vs: vs.reshape(-1, 3).tolist())
        df_cache['centroid'] = df_cache['centroid'].apply(lambda c: c.tolist())
        df_cache['normal'] = df_cache['normal'].apply(lambda n: n.tolist())
        df_cache.to_parquet(parquet_file)
        
        return df
    except Exception as e:
        print(f"加载数据失败: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()

def step4_generate_mapping():
    try:
        print("步骤4: 生成映射对照表...")
        start_time = time.time()
        
        if os.path.exists(MAP_XLSX) and not FORCE_REBUILD:
            print(f"检测到映射表文件已存在: {MAP_XLSX}")
            print("跳过步骤4计算")
            return MAP_XLSX
        
        panel_df = load_data(PANEL_XLSX)
        struct_df = load_data(STRUCTURE_XLSX)
        
        if panel_df.empty or struct_df.empty:
            raise Exception("面板数据或结构数据为空")
        
        panel_centers = np.stack(panel_df.centroid.values)
        panel_norms = np.stack(panel_df.normal.values)
        panel_areas = panel_df.area.values
        panel_ids = panel_df.e_num.values
        panel_edges = [np.array([[p1,p2] for p1,p2 in zip(vs, np.roll(vs, -1, axis=0))]) for vs in panel_df.vertices.values]
        
        struct_centers = np.stack(struct_df.centroid.values)
        struct_norms = np.stack(struct_df.normal.values)
        
        kdtree = KDTree(panel_centers)

        result_panel_ids = np.full(len(struct_df), np.nan)

        distol = DISTOL / 100
        
        for i in tqdm(range(len(struct_df)), desc="ID映射"):
            sc = struct_centers[i]
            if sc[2] > Z_THRESHOLD: 
                continue
                
            distances, candidates = kdtree.query(sc, k=MAX_CANDIDATES)
            sn = struct_norms[i]
            
            for idx in candidates:
                pn = panel_norms[idx]
                cos_theta = abs(np.dot(sn, pn))
                angle = np.degrees(np.arccos(np.clip(cos_theta, 0.0, 1.0)))

                # 计算阴影面积比例
                AΣ = compute_shadow_area(sc, panel_edges[idx])
                area_ratio = AΣ / panel_areas[idx]

                if angle > ANGTOL or abs(area_ratio - 1) > (distol):
                    continue
                    
                result_panel_ids[i] = panel_ids[idx]
                break
        
        result_df = pd.DataFrame({
            'struct_id': struct_df['e_num'].values,
            'mapped_panel_id': result_panel_ids
        })
        result_df.to_excel(MAP_XLSX, index=False, float_format='%.5f')
        
        end_time = time.time()
        print(f"步骤4完成，耗时: {end_time - start_time:.2f}秒")
        return MAP_XLSX
    except Exception as e:
        print(f"步骤4失败: {str(e)}")
        traceback.print_exc()
        return None

def parse_t_file_for_coords(t_file_path):
    node_coords = {}
    with open(t_file_path, 'r') as file:
        for line in file:
            if line.startswith('GCOORD'):
                parts = line.split()
                try:
                    node = float(parts[1])
                    x = float(parts[2]) * 1000
                    y = float(parts[3]) * 1000
                    z = float(parts[4]) * 1000
                    node_coords[node] = (x, y, z)
                except (IndexError, ValueError) as e:
                    pass
    return node_coords

def process_other_data(brigac_df, bgrav_df, bnload_df, case_id, node_coords_map, global_special_node_id, all_bnload_nodes):
    other_row = {'time': case_id}

    brigac_data = brigac_df[brigac_df['工况'].astype(str) == str(case_id)]
    bgrav_data = bgrav_df[bgrav_df['工况'].astype(str) == str(case_id)]
    
    if not brigac_data.empty and not bgrav_data.empty:
        brigac_row = brigac_data.iloc[0]
        bgrav_row = bgrav_data.iloc[0]
        other_row['ACCEx'] = brigac_row['X平移加速度'] - bgrav_row['重力X分量']
        other_row['ACCEy'] = brigac_row['Y平移加速度'] - bgrav_row['重力Y分量']
        other_row['ACCEz'] = brigac_row['Z平移加速度'] - bgrav_row['重力Z分量']
        other_row['ACCErx'] = brigac_row['X旋转加速度']
        other_row['ACCEry'] = brigac_row['Y旋转加速度']
        other_row['ACCErz'] = brigac_row['Z旋转加速度']
    else:
        other_row.update({
            'ACCEx': 0.0, 'ACCEy': 0.0, 'ACCEz': 0.0,
            'ACCErx': 0.0, 'ACCEry': 0.0, 'ACCErz': 0.0
        })
    
    current_special_node_id = global_special_node_id
    
    if not bnload_df.empty:
        if current_special_node_id is None:
            for _, row in bnload_df.iterrows():
                if (row['自由度1'] != 0.0 and row['自由度2'] != 0.0 and row['自由度3'] != 0.0 and
                    row['自由度4'] != 0.0 and row['自由度5'] != 0.0 and row['自由度6'] != 0.0):
                    current_special_node_id = row['节点编号']
                    other_row['special_node_id'] = current_special_node_id
                    break
        
        special_node_data = None
        if current_special_node_id is not None:
            special_node_data = bnload_df[bnload_df['节点编号'] == current_special_node_id]
            if not special_node_data.empty:
                special_node_data = special_node_data.iloc[0]
        
        if special_node_data is not None:
            other_row.update({
                'TwrBsFxt': special_node_data['自由度1'],
                'TwrBsFyt': special_node_data['自由度2'],
                'TwrBsFzt': special_node_data['自由度3'],
                'TwrBsMxt': special_node_data['自由度4'],
                'TwrBsMyt': special_node_data['自由度5'],
                'TwrBsMzt': special_node_data['自由度6']
            })
        else:
            other_row.update({
                'TwrBsFxt': 0.0, 'TwrBsFyt': 0.0, 'TwrBsFzt': 0.0,
                'TwrBsMxt': 0.0, 'TwrBsMyt': 0.0, 'TwrBsMzt': 0.0
            })
        
        if not all_bnload_nodes:
            if current_special_node_id is not None:
                all_bnload_nodes.append(current_special_node_id)
                
            for _, row in bnload_df.iterrows():
                node_id = row['节点编号']
                if current_special_node_id is None or node_id != current_special_node_id:
                    if node_id not in all_bnload_nodes:
                        all_bnload_nodes.append(node_id)
        
        for idx, node_id in enumerate(all_bnload_nodes):
            if node_id == current_special_node_id:
                continue
                
            node_data = bnload_df[bnload_df['节点编号'] == node_id]
            if not node_data.empty:
                node_row = node_data.iloc[0]
                line_index = idx
                if current_special_node_id is not None and current_special_node_id in all_bnload_nodes:
                    if idx == 0:
                        continue
                    line_index = idx
                
                other_row[f'LINE{line_index}-X'] = node_row['自由度1']
                other_row[f'LINE{line_index}-Y'] = node_row['自由度2']
                other_row[f'LINE{line_index}-Z'] = node_row['自由度3']
            else:
                other_row[f'LINE{line_index}-X'] = 0.0
                other_row[f'LINE{line_index}-Y'] = 0.0
                other_row[f'LINE{line_index}-Z'] = 0.0
    else:
        other_row.update({
            'TwrBsFxt': 0.0, 'TwrBsFyt': 0.0, 'TwrBsFzt': 0.0,
            'TwrBsMxt': 0.0, 'TwrBsMyt': 0.0, 'TwrBsMzt': 0.0
        })
        
        for idx in range(len(all_bnload_nodes)):
            if idx == 0 and current_special_node_id is not None:
                continue
            other_row[f'LINE{idx}-X'] = 0.0
            other_row[f'LINE{idx}-Y'] = 0.0
            other_row[f'LINE{idx}-Z'] = 0.0
    
    return other_row

def split_and_process_all_conditions(l_file, t_file, map_xlsx, structure_xlsx, output_dir, mode="new"):
    try:
        file_id = os.path.basename(l_file).split('.')[0]
        print(f"处理文件: {file_id}, 模式: {mode}")
        start_time = time.time()
        
        result_dir = os.path.join(output_dir, f"results_{file_id}")
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir, exist_ok=True)
        
        mapper = PressureMapper(map_xlsx, structure_xlsx)
        processor = BlockProcessor()
        
        print(f"开始拆分工况文件: {l_file}")
        with open(l_file, 'r', buffering=BUFFER_SIZE) as f:
            for line in tqdm(f, desc=f"处理 {file_id}"):
                processor.process_line(line)
        
        processor.finalize_block()
        all_conditions = processor.get_all_conditions()
        print(f"文件 {file_id} 发现的工况总数: {len(all_conditions)}")
        
        processed_count = 0
        pressure_data_list = []  # 存储每个工况的压力数据
        all_other_data = []      # 存储其他载荷数据
        bello2_data_dict = {}     # 存储每个工况的BELLO2数据
        node_coords_map = None   # 节点坐标映射
        global_special_node_id = None  # 存储全局特殊节点的ID（不是载荷值）
        all_bnload_nodes = []    # 存储所有节点顺序
        
        # 获取节点坐标映射（仅一次）
        if mode == "new":
            node_coords_map = parse_t_file_for_coords(t_file)
        
        # 使用单个进度条显示所有工况的处理进度
        for condition in tqdm(all_conditions, desc=f"处理工况 {file_id}", total=len(all_conditions)):
            try:
                condition_str = processor.get_condition_data(condition)
                output_excel = os.path.join(result_dir, f"result_{condition}.xlsx")
                
                # 解析压力数据
                panel_df = mapper.parse_beuslo(io.StringIO(condition_str))
                pressure_df = mapper.map_pressure(panel_df)
                
                # 确保e_num是浮点数类型
                if not pressure_df.empty:
                    pressure_df['e_num'] = pressure_df['e_num'].astype(float).round(5)
                
                # 解析其他数据
                brigac_df = mapper.extract_brigac_str(condition_str)
                bgrav_df = mapper.extract_bgrav_str(condition_str)
                bnload_df = mapper.process_bnload_str(condition_str, t_file)
                bello2_df = mapper.process_bello2_str(condition_str, t_file)
                
                # 添加到相应的数据结构中
                if mode == "new":
                    # 准备收集数据（不立即合并）
                    pressure_case_df = pressure_df.rename(columns={'new_pressure': condition})
                    pressure_data_list.append(pressure_case_df[['e_num', condition]])
                    
                    # 保存BELLO2数据
                    bello2_data_dict[condition] = bello2_df
                    
                    # 处理其他载荷数据
                    other_row = process_other_data(
                        brigac_df, 
                        bgrav_df, 
                        bnload_df, 
                        condition,
                        node_coords_map,
                        global_special_node_id,  # 传递节点ID
                        all_bnload_nodes
                    )
                    
                    # 更新全局特殊节点ID（仅首次设置）
                    if 'special_node_id' in other_row and global_special_node_id is None:
                        global_special_node_id = other_row['special_node_id']
                        del other_row['special_node_id']  # 从行中移除特殊字段
                    
                    all_other_data.append(other_row)
                
                # ORIGINAL模式：为当前工况创建Excel文件
                else:
                    # 创建工况Excel文件
                    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                        # 1. Pressure Data
                        pressure_df[['e_num', 'new_pressure']].to_excel(
                            writer, 
                            sheet_name="Pressure Data", 
                            index=False,
                            float_format='%.5f'
                        )
                        
                        # 2. BRIGAC Data
                        if not brigac_df.empty:
                            brigac_df.to_excel(
                                writer, 
                                sheet_name="BRIGAC Data", 
                                index=False,
                                float_format='%.5f'
                            )
                        else:
                            # 创建空工作表
                            pd.DataFrame({'信息': ['无数据']}).to_excel(
                                writer, 
                                sheet_name="BRIGAC Data", 
                                index=False
                            )
                        
                        # 3. BGRAV Data
                        if not bgrav_df.empty:
                            bgrav_df.to_excel(
                                writer, 
                                sheet_name="BGRAV Data", 
                                index=False,
                                float_format='%.5f'
                            )
                        else:
                            pd.DataFrame({'信息': ['无数据']}).to_excel(
                                writer, 
                                sheet_name="BGRAV Data", 
                                index=False
                            )
                        
                        # 4. BNLOAD Data
                        if not bnload_df.empty:
                            bnload_df.to_excel(
                                writer, 
                                sheet_name="BNLOAD Data", 
                                index=False,
                                float_format='%.5f'
                            )
                        else:
                            pd.DataFrame({'信息': ['无数据']}).to_excel(
                                writer, 
                                sheet_name="BNLOAD Data", 
                                index=False
                            )
                        
                        # 5. BELLO2 Data
                        if not bello2_df.empty:
                            bello2_df.to_excel(
                                writer, 
                                sheet_name="BELLO2 Data", 
                                index=False,
                                float_format='%.5f'
                            )
                        else:
                            pd.DataFrame({'信息': ['无数据']}).to_excel(
                                writer, 
                                sheet_name="BELLO2 Data", 
                                index=False
                            )
                
                processed_count += 1
                
            except Exception as e:
                print(f"处理工况 {condition} 失败: {str(e)}")
                traceback.print_exc()
                processed_count += 1
        
        # 新模式：保存所有收集的数据
        if mode == "new":
            print(f"保存 {file_id} 的新模式结果...")
            
            # 1. 合并压力数据（优化部分）
            if pressure_data_list:
                # 获取所有工况的载荷值（按单元顺序直接堆叠）
                all_case_ids = [df.columns[1] for df in pressure_data_list]
                pressure_values = [df.iloc[:, 1].values for df in pressure_data_list]
                
                # 优化点：使用字典一次性创建DataFrame
                # 创建列数据字典
                columns_dict = {'e_num': pressure_data_list[0]['e_num'].values}
                
                # 添加所有工况数据
                for i, case_id in enumerate(all_case_ids):
                    columns_dict[case_id] = pressure_values[i]
                
                # 一次性创建DataFrame
                pressure_wide = pd.DataFrame(columns_dict)
                
                # 保存压力数据
                pressure_output = os.path.join(result_dir, f"{file_id}_pressures.csv")
                pressure_wide.to_csv(pressure_output, index=False, float_format='%.5f')
            
            # 2. 合并其他数据
            if all_other_data:
                other_df = pd.DataFrame(all_other_data)
                # 确保time列是浮点数并保留五位小数
                other_df['time'] = other_df['time'].astype(float).round(5)
                other_output = os.path.join(result_dir, f"{file_id}_other_data.csv")
                other_df.to_csv(other_output, index=False, float_format='%.5f')
            
            # 3. 合并BELLO2数据（优化部分）
            if bello2_data_dict:
                # 收集所有单元和坐标信息
                first_case = next(iter(bello2_data_dict.values()))
                bello2_base = first_case[['单元编号', '起点X', '起点Y', '起点Z', '终点X', '终点Y', '终点Z']].copy()
                bello2_base.rename(columns={'单元编号': 'line_num'}, inplace=True)
                bello2_base.rename(columns={'起点X': 'startX'}, inplace=True)
                bello2_base.rename(columns={'起点Y': 'startY'}, inplace=True)
                bello2_base.rename(columns={'起点Z': 'startZ'}, inplace=True)
                bello2_base.rename(columns={'终点X': 'endX'}, inplace=True)
                bello2_base.rename(columns={'终点Y': 'endY'}, inplace=True)
                bello2_base.rename(columns={'终点Z': 'endZ'}, inplace=True)
                
                # 确保line_num是浮点数并保留五位小数
                bello2_base['line_num'] = bello2_base['line_num'].astype(float).round(5)
                
                # 优化点：使用字典收集所有载荷列
                bello_data_to_add = {}
                
                # 添加载荷数据
                for case_id, df in bello2_data_dict.items():
                    bello_data_to_add[f'Fx_{case_id}'] = df['自由度1载荷'].values
                    bello_data_to_add[f'Fy_{case_id}'] = df['自由度2载荷'].values
                    bello_data_to_add[f'Fz_{case_id}'] = df['自由度3载荷'].values
                
                # 一次性添加所有载荷列
                bello2_base = pd.concat([bello2_base, pd.DataFrame(bello_data_to_add)], axis=1)
                
                # 保存BELLO2数据
                bello2_output = os.path.join(result_dir, f"{file_id}_bello2_data.csv")
                bello2_base.to_csv(bello2_output, index=False, float_format='%.5f')
            
            # 4. 保存节点坐标（只保存一次）
            if all_bnload_nodes and node_coords_map:
                coords_list = [node_coords_map.get(node_id, (0.0, 0.0, 0.0)) for node_id in all_bnload_nodes]
                coord_df = pd.DataFrame(coords_list, columns=['x', 'y', 'z'])
                coord_output = os.path.join(result_dir, f"{file_id}_node_coords.csv")
                coord_df.to_csv(coord_output, index=False, float_format='%.5f')
        
        end_time = time.time()
        print(f"文件 {file_id} 处理完成: {processed_count}/{len(all_conditions)} 个工况, 耗时: {end_time - start_time:.2f}秒")
        return True
    except Exception as e:
        print(f"文件 {file_id} 处理失败: {str(e)}")
        traceback.print_exc()
        return False

def process_single_l_file(l_file, t_file, map_xlsx, structure_xlsx, output_dir, mode):
    """处理单个L文件的包装函数"""
    try:
        # 为每个进程设置唯一标识
        pid = os.getpid()
        print(f"进程 {pid} 开始处理文件: {l_file}")
        start_time = time.time()
        
        # 调用处理函数
        result = split_and_process_all_conditions(
            l_file=l_file,
            t_file=t_file,
            map_xlsx=map_xlsx,
            structure_xlsx=structure_xlsx,
            output_dir=output_dir,
            mode=mode
        )
        
        end_time = time.time()
        print(f"进程 {pid} 完成处理文件: {l_file}, 耗时: {end_time - start_time:.2f}秒")
        return result
    except Exception as e:
        print(f"处理文件 {l_file} 时发生未捕获异常: {str(e)}")
        traceback.print_exc()
        return False

# ========================
# 主执行流程
# ========================
def main():
    try:
        print("开始执行完整载荷提取流程...")
        print("=====================================")
        total_start_time = time.time()
        
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 检查所有结果文件是否存在
        results_exist = all([
            os.path.exists(STRUCTURE_XLSX),
            os.path.exists(PANEL_XLSX),
            os.path.exists(MAP_XLSX)
        ]) and not FORCE_REBUILD
        
        # 步骤1：预处理东大模型（仅在需要时执行）
        if not results_exist or not os.path.exists(STRUCTURE_XLSX):
            step1_time_start = time.time()
            if not step1_preprocess():
                print("步骤1失败，终止流程")
                return
            step1_time_end = time.time()
            print(f"步骤1总耗时: {step1_time_end - step1_time_start:.2f}秒")
            print("=====================================")
        else:
            print(f"检测到结构数据文件已存在: {STRUCTURE_XLSX}")
            print("跳过步骤1计算")
            print("=====================================")
        
        # 步骤2和3：仅在没有结果文件时执行
        if not results_exist or not os.path.exists(PANEL_XLSX):
            # 使用第一个L文件执行步骤2和3
            if L_FILES:
                first_l_file = L_FILES[0]
                print(f"使用第一个L文件执行步骤2和3: {first_l_file}")
                
                # 步骤2：提取第一个工况
                step2_time_start = time.time()
                first_condition_file = step2_extract_first_condition(first_l_file)
                if not first_condition_file:
                    print("步骤2失败，终止流程")
                    return
                step2_time_end = time.time()
                print(f"步骤2总耗时: {step2_time_end - step2_time_start:.2f}秒")
                print("=====================================")
                
                # 步骤3：提取面压力
                step3_time_start = time.time()
                if not step3_extract_pressure(first_condition_file, T_FILE):
                    print("步骤3失败，终止流程")
                    return
                step3_time_end = time.time()
                print(f"步骤3总耗时: {step3_time_end - step3_time_start:.2f}秒")
                print("=====================================")
            else:
                print("错误: 未提供L文件列表")
                return
        else:
            print(f"检测到面板数据文件已存在: {PANEL_XLSX}")
            print("跳过步骤2和3计算")
            print("=====================================")
        
        # 步骤4：生成映射表（仅在需要时执行）
        if not results_exist or not os.path.exists(MAP_XLSX):
            step4_time_start = time.time()
            if not step4_generate_mapping():
                print("步骤4失败，终止流程")
                return
            step4_time_end = time.time()
            print(f"步骤4总耗时: {step4_time_end - step4_time_start:.2f}秒")
            print("=====================================")
        else:
            print(f"检测到映射表文件已存在: {MAP_XLSX}")
            print("跳过步骤4计算")
            print("=====================================")
        
        # 步骤5&6：并行处理所有L文件
        print(f"开始并行处理 {len(L_FILES)} 个L文件...")
        step56_time_start = time.time()
        
        # 使用进程池并行处理
        max_workers = min(MAX_WORKERS, len(L_FILES))  # 限制最大并行进程数
        success_count = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for l_file in L_FILES:
                future = executor.submit(
                    process_single_l_file,
                    l_file=l_file,
                    t_file=T_FILE,
                    map_xlsx=MAP_XLSX,
                    structure_xlsx=STRUCTURE_XLSX,
                    output_dir=OUTPUT_DIR,
                    mode=OUTPUT_MODE
                )
                futures[future] = l_file
            
            # 监控进度并处理结果
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理L文件"):
                l_file = futures[future]
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                        print(f"文件 {l_file} 处理成功")
                    else:
                        print(f"文件 {l_file} 处理失败")
                except Exception as e:
                    print(f"处理文件 {l_file} 时发生异常: {str(e)}")
                    traceback.print_exc()
        
        step56_time_end = time.time()
        print(f"L文件处理完成: {success_count}/{len(L_FILES)} 个文件成功, 总耗时: {step56_time_end - step56_time_start:.2f}秒")
        print("=====================================")
        
        total_end_time = time.time()
        print(f"完整载荷提取流程执行完毕！总耗时: {total_end_time - total_start_time:.2f}秒")
    except Exception as e:
        print(f"主流程执行失败: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # 设置多进程启动方法
    multiprocessing.set_start_method('spawn', force=True)
    main()