from .pretokenization import pretokenize


def train(file_path: str, num_processes: int) -> list[str]:
    frequency_table = pretokenize(file_path, num_processes)

    return frequency_table
