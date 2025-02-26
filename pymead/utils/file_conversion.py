import subprocess as sp
import os
import shutil

from pymead import DependencyNotFoundError


def convert_ps_to_pdf(conversion_dir: str, input_file_name: str, output_file_name: str, timeout=10.0):
    if shutil.which('ps2pdf'):
        proc = sp.Popen(['ps2pdf', input_file_name, output_file_name], stdout=sp.PIPE, stderr=sp.PIPE,
                        cwd=conversion_dir, shell=False)
        log_file_name = 'ps2pdf.log'
        log_file = os.path.join(conversion_dir, log_file_name)
        ps2pdf_complete = False
        with open(log_file, 'wb') as f:
            try:
                outs, errs = proc.communicate(timeout=timeout)
                f.write('Output:\n'.encode('utf-8'))
                f.write(outs)
                f.write('\nErrors:\n'.encode('utf-8'))
                f.write(errs)
                ps2pdf_complete = True
            except sp.TimeoutExpired:
                proc.kill()
                outs, errs = proc.communicate()
                f.write('After timeout, \nOutput: \n'.encode('utf-8'))
                f.write(outs)
                f.write('\nErrors:\n'.encode('utf-8'))
                f.write(errs)
    else:
        raise DependencyNotFoundError(
            "Ghostscript ps2pdf tool (executable or batch file) not found on system path. See "
            "https://pymead.readthedocs.io/en/latest/install.html#optional for a link to the Ghostscript page. "
            "If you already completed the installation and added the path to the executable or directory "
            "containing the executable to the system path, you may need to restart your terminal or IDE "
            "for the changes to apply.")
    return ps2pdf_complete, log_file


def convert_pdf_to_svg(conversion_dir: str, input_file_name: str, output_file_name: str, timeout=10.0):
    if shutil.which('mutool'):
        proc = sp.Popen(['mutool', 'convert', '-o', output_file_name, input_file_name], stdout=sp.PIPE, stderr=sp.PIPE,
                        cwd=conversion_dir, shell=False)
        log_file_name = 'mutool.log'
        log_file = os.path.join(conversion_dir, log_file_name)
        mutool_complete = False
        with open(log_file, 'wb') as f:
            try:
                outs, errs = proc.communicate(timeout=timeout)
                f.write('Output:\n'.encode('utf-8'))
                f.write(outs)
                f.write('\nErrors:\n'.encode('utf-8'))
                f.write(errs)
                mutool_complete = True
            except sp.TimeoutExpired:
                proc.kill()
                outs, errs = proc.communicate()
                f.write('After timeout, \nOutput: \n'.encode('utf-8'))
                f.write(outs)
                f.write('\nErrors:\n'.encode('utf-8'))
                f.write(errs)
    else:
        raise DependencyNotFoundError(
            "MuPDF mutool executable not found on system path. See "
            "https://pymead.readthedocs.io/en/latest/install.html#optional for a link to the MuPDF page. "
            "If you already completed the installation and added the path to the executable or directory "
            "containing the executable to the system path, you may need to restart pymead "
            "for the changes to apply.")
    return mutool_complete, log_file


def convert_ps_to_svg(conversion_dir: str, input_file_name: str, intermediate_pdf_file_name: str,
                      output_file_name: str, timeout=10.0):
    ps2pdf_complete, ps2pdf_log_file = convert_ps_to_pdf(conversion_dir, input_file_name, intermediate_pdf_file_name,
                                                         timeout=timeout)
    with open(ps2pdf_log_file, "r") as plfile:
        lines = plfile.readlines()
    for line in lines:
        print(f"ps2pdf log {line = }")
    if ps2pdf_complete:
        mutool_complete, mutool_log_file = convert_pdf_to_svg(conversion_dir, intermediate_pdf_file_name,
                                                              output_file_name, timeout=timeout)
        if mutool_complete:
            split_path = os.path.splitext(output_file_name)
            os.replace(os.path.join(conversion_dir, f"{split_path[0]}1{split_path[-1]}"),
                       os.path.join(conversion_dir, output_file_name))
            return True, {'ps2pdf': ps2pdf_log_file, 'mutool': mutool_log_file}
        else:
            return False, {'ps2pdf': ps2pdf_log_file, 'mutool': mutool_log_file}
    else:
        return False, {'ps2pdf': ps2pdf_log_file, 'mutool': ''}


class FileConversionError(Exception):
    pass

