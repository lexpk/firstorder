from .tptp import from_tptp_file, to_tptp_file
from .tptp import from_tptp_proof_string
from subprocess import check_output


def solve_tptp(
    file: str,
    include_dir: str = "",
    timeout: int = 10,
):
    problem = from_tptp_file(file, include_dir)
    to_tptp_file(problem, '/tmp/firstorder.p')
    output = check_output(['/root/provers/vampire/vampire_z3_rel_static_HEAD_6295',
                          '--proof', 'proofcheck', '--time_limit', str(timeout), '/tmp/firstorder.p'])
    return from_tptp_proof_string(output.decode('utf-8'))
