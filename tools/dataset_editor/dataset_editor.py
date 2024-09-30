import os
import shutil
import random
from pathlib import Path
from utils.io import *

def move_file_by_txt(txt_file_path, src_folder, target_folder, mode='move', suffix=None):
    """
    Given a .txt file, move all listed files (if exist) from one folder to another folder.
    
    Args:
        txt_file_path (str): .txt file that includes all the files (could be images or labels) to move
        src_folder (str): original file path
        target_folder (str): target file path
        mode (str): either move or copy [Default: 'move']
        suffix (str): needed if files in txt_file_path do not have suffix [Default: None]

    Return:
        None
    """

    if not (txt_file_path and src_folder and target_folder):
        print(Error('Not enough arguments provided!'))
        exit()

    if mode != 'copy' and mode != 'move':
        print(Error('Please use a valid mode: <move> or <copy>'))
        exit()
    
    txt_file_path, src_folder, target_folder = Path(txt_file_path), Path(src_folder), Path(target_folder)
    if not (os.path.exists(txt_file_path) and os.path.exists(src_folder) and os.path.exists(target_folder)):
        print(Error('One or more invalid path detected, please enter correct path.'))
        exit()
    
    files = []
    with open(txt_file_path, 'r') as file:
        for line in file:
            file_name = line.strip()
            
            if len(Path(file_name).suffix) < 1 and not suffix:
                print(Error(f'the file {file_name} does not have any suffix, ' \
                               'and no suffix recieved in args'))
                exit()
                
            if len(Path(file_name).suffix) < 1 and suffix:
                files.append(file_name + '.' + suffix.lstrip('.'))
            else:
                files.append(file_name)
    
    progress = Progress()
    moved_count = 0
    total = len(files)
    for file in files:
        file_path = src_folder / file
        if os.path.exists(file_path):
            moved_count += 1
            action = 'moving' if mode == 'move' else 'copying'
            progress.progress_bar(moved_count, total, Info(f'{action}: {str(file_path)}'))
            if mode == 'copy':
                shutil.copy2(file_path, target_folder)
            elif mode == 'move':
                shutil.move(file_path, target_folder)
        else:
            print(Warning(f'\"{file_path}\" does not exist'))
                
    action = 'moved' if mode == 'move' else 'copied'
    print(Info(f"Done! {moved_count} files are {action}."))

def select_and_move_files(source_folder, target_folder, num_files=None, percantage=None, output_file='choosed_files.txt'):
    """
    从指定文件夹中随机选取一定数量的文件并移动到目标文件夹。
    
    参数:
    source_folder (str): 源文件夹路径。
    target_folder (str): 目标文件夹路径。
    num_files (int): 要随机选取并移动的文件数量。
    """
    # 获取源文件夹中所有文件的完整路径
    all_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    if num_files and percantage:
        print(Error("Cannot handle both num_files and percantage."))
        exit()
    
    # 如果文件数小于要选取的数量，返回全部文件
    if len(all_files) <= num_files:
        selected_files = all_files
    else:
        # 随机选取指定数量的文件
        selected_files = random.sample(all_files, num_files)
    
    # 检查目标文件夹是否存在，不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 存储选中文件的无后缀文件名
    choosed_file_names = []

    # 移动选中的文件到目标文件夹
    for file_path in selected_files:
        file_name = os.path.basename(file_path)
        choosed_file_names.append(os.path.splitext(file_name)[0])
        shutil.move(file_path, os.path.join(target_folder, file_name))
        print(Info(f"文件 {file_name} 已移动到 {target_folder}"))
    
    # 将选中的文件名（无后缀）保存到txt文件中
    with open(output_file, 'w') as f:
        for file_name in choosed_file_names:
            f.write(file_name + '\n')

    return selected_files
     
def temp():
    pass

def main():
    print(Info('test'))
