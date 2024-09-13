import os
import shutil
from pathlib import Path
from utils.io import *

def move_file_by_txt(txt_file_path, src_folder, target_folder, mode='move'):
    """
    Given a .txt file, move all listed files (if exist) from one folder to another folder.
    
    Args:
        txt_file_path (str): .txt file that includes all the files (could be images or labels) to move
        src_folder (str): original file path
        target_folder (str): target file path
        mode (str): either move or copy Default: move

    Return:
        None
    """
    suffix = ['jpg', 'jpeg', 'png']

    if not (txt_file_path and src_folder and target_folder):
        print(Error('Not enough arguments provided!'))
        return -1

    if mode != 'copy' and mode != 'move':
        print(Error('Please use a valid mode: <move> or <copy>'))
        return -1
    
    txt_file_path, src_folder, target_folder = Path(txt_file_path), Path(src_folder), Path(target_folder)
    if not (os.path.exists(txt_file_path) and os.path.exists(src_folder) and os.path.exists(target_folder)):
        print(Error('One or more invalid path detected, please enter correct path.'))
        return -1
    
    files = []
    with open(txt_file_path, 'r') as file:
        for line in file:
            file_name = line.strip()
            
            if len(Path(file_name).suffix) < 1:
                print(Warning(f'the file {file_name} does not have any suffix, ' \
                               'moving process will continue based on .jpg image name only'))
                
                file_name += '.jpg'
            elif Path(file_name).suffix not in suffix:
                print(Warning(f'the file {file_name} is skipped since it does ' \
                               f'not have a valid suffix: {suffix}'))
                continue

            if file_name:
                files.append(file_name)
    
    progress = Progress()
    current = 0
    total = len(files)
    for file in files:
        file_path = src_folder / file
        if os.path.exists(file_path):
            action = 'moving' if mode == 'move' else 'copying'
            #print(Info(f'{action}: {str(file_path)}'), flush=True)
            progress.progress_bar(current, total, Info(f'{action}: {str(file_path)}'))
            current += 1
            if mode == 'copy':
                shutil.copy2(file_path, target_folder)
            elif mode == 'move':
                shutil.move(file_path, target_folder)
                
    action = 'moved' if mode == 'move' else 'copied'
    print(Info(f"Done! {len(files)} files are {action}."))

     
def move_label():
    pass
