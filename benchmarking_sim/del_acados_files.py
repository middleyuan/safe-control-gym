import os

current_dir = os.getcwd()
print(current_dir)

# delete the folder has *_del_*
for root, dirs, files in os.walk(current_dir):
    for dir in dirs:
        if 'c_generated_code' in dir:
            os.system(f'rm -rf {os.path.join(root, dir)}')
            print('delete:', os.path.join(root, dir))
    for files in files:
        if 'acados_ocp_solver' in files:
            os.system(f'rm -rf {os.path.join(root, files)}')
            print('delete:', os.path.join(root, files))
