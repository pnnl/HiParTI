#!/usr/bin/env python3

import logging
import re
import subprocess
import os

tensors_to_test = [
    '/home/sbliss/tensors/20^3.tns',                    # 8002
    '/opt/data/jli/BIGTENSORS/nips-4d.tns',             # 3101611
    '/home/sbliss/tensors/choa100k.tns',                # 4142859
    '/opt/data/jli/BIGTENSORS/choa700k.tns',            # 26953734
    '/opt/data/jli/BIGTENSORS/enron-4d.tns',            # 54202101
    '/opt/data/jli/BIGTENSORS/nell2.tns',               # 76879421
    '/opt/data/jli/BIGTENSORS/freebase_music.tns',      # 99546553
    '/opt/data/jli/BIGTENSORS/freebase_sampled.tns',    # 139920773
    '/opt/data/jli/BIGTENSORS/delicious.tns',           # 140126183
    '/opt/data/jli/BIGTENSORS/nell1.tns',               # 143599554
    '/opt/data/jli/BIGTENSORS/uber-4d.tns',             #
    '/opt/data/jli/BIGTENSORS/flickr-4d.tns',           #
    '/opt/data/jli/BIGTENSORS/delicious-4d.tns',        #
]

dense_tensor_generator = '/home/sbliss/ParTImm/tools/generate_dense_tensor.py'
temp_tensor1 = '/tmp/temp_dense_tensor_1.tns'
temp_tensor2 = '/tmp/temp_dense_tensor_2.tns'
temp_tensor3 = '/tmp/temp_sparse_tensor.tns'
old_test_program = '/home/sbliss/ParTI/build/tests/test_ttm_speed'
new_test_program = '/home/sbliss/ParTImm/build/tests/test_ttm_chain_speed'

R_value = 16
old_cuda_device = 0
new_cuda_device = 8

def main():
    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)
    hlog = logging.FileHandler('test_ssttm.log', encoding='utf-8')
    hlog.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
    logging.getLogger().addHandler(hlog)
    logging.info(':: Start test')

    report = open('test_ssttm.csv', 'w', encoding='utf-8')
    report.write(','.join(['tensor'] + ['new' + str(i) for i in range(-3, 5)] + ['old' + str(i) for i in range(-2, 5)] + ['old_avg', 'new_avg']) + '\n')

    row = 2
    for tensor in tensors_to_test:
        f = open(tensor, 'r')
        num_modes = int(f.readline())
        shape = list(map(int, f.readline().split()))
        f.close()
        logging.info('Tensor: {}'.format(tensor))
        report.write('"{}"'.format(tensor))

        cmdline = [dense_tensor_generator, temp_tensor1, str(R_value), str(shape[0])]
        logging.info(str(cmdline))
        subprocess.run(cmdline)

        cmdline = [dense_tensor_generator, temp_tensor2, str(R_value), str(shape[1])]
        logging.info(str(cmdline))
        subprocess.run(cmdline)

        cmdline = [new_test_program, tensor, temp_tensor1, temp_tensor2, temp_tensor3, '--dev', str(new_cuda_device)]
        logging.info(str(cmdline))
        with subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True) as proc:
            for line in proc.stdout:
                logging.info(line.rstrip())
                match = re.match(r'\[.* TTM Kernel\]: (.*) s spent on device ', line)
                if match:
                    report.write(',' + match.group(1))
                    report.flush()

        cmdline = [old_test_program, temp_tensor3, temp_tensor2, '1', str(old_cuda_device)]
        logging.info(str(cmdline))
        with subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True) as proc:
            for line in proc.stdout:
                logging.info(line.rstrip())
                match = re.match(r'\[.* SpTns \* Mtx\]: (.*) s$', line)
                if match:
                    report.write(',' + match.group(1))
                    report.flush()

        try:
            os.unlink(temp_tensor1)
        except FileNotFoundError:
            pass
        try:
            os.unlink(temp_tensor2)
        except FileNotFoundError:
            pass
        try:
            os.unlink(temp_tensor3)
        except FileNotFoundError:
            pass

        report.write(',=AVERAGE(L{}:P{}),=AVERAGE(E{}:I{})\n'.format(row, row, row, row))
        report.flush()
        row += 1

    report.close()
    logging.info(':: Finish test')


if __name__ == '__main__':
    main()
