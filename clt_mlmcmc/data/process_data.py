import re
import numpy as np
from numbers import Number

variables = ['h0', 'h0u', 'h0v', 'b', 'h1', 'h1u', 'h1v', 'z']
n = len(variables)


def parse_patch_block(patch_text, unknown=None):
    offset_match = re.search(r"offset\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)", patch_text)
    values_match = re.search(r'begin cell-values\s+".*?"\s+(.*?)\s+end cell-values', patch_text, re.DOTALL)
    if not offset_match or not values_match:
        return None

    offset = (float(offset_match.group(1)), float(offset_match.group(2)))
    values_str = values_match.group(1)
    values = list(map(float, values_str.strip().split()))
    if len(values) % len(variables) != 0:
        raise ValueError("Check formatting.")
    
    
    if not unknown:
        values = dict()
        for i in range(n):
            if variables[i] in unknown:
                values[variables[i]] = values[i::n]
        return offset, values
    else:
        return offset, {variables[i]: values[i::n] for i in range(n)}


def parse_patch_file(filename, unknown=None, folder="sim_files"):
    file = "./" + folder + "/" + filename
    with open(file, "r") as f:
        contents = f.read()
    
    if "begin patch" not in contents:
        raise ValueError("No patches found in file.")
    patches = contents.split("begin patch", 1)[1]
    patch_blocks = patches.split("begin patch")
    all_patches = {}

    for i, patch_body in enumerate(patch_blocks):
        patch_text = "begin_patch" + patch_body
        if "end patch" not in patch_text:
            continue
        patch_text = patch_text.split("end patch")[0] + "end patch"
        parsed = parse_patch_block(patch_text, unknown=unknown)
        if parsed:
            offset, data = parsed
            if offset in all_patches:
                print(f" Duplicate offset {offset} in patch {i}")
            all_patches[offset] = data
    
    return all_patches


def parse_snapshot(files, unknown=None, folder="sim_files"):
    full_snapshot = {}
    for file in files:
        current_file = parse_patch_file(file, unknown=unknown, folder=folder)
        # maybe it's worth checking for repeat offsets? but shouldn't be a problem
        full_snapshot |= current_file
    return full_snapshot


def postprocessing(filename, snapshot='last', unknown=None, folder="sim_files"):
    with open(filename, "r") as f:
        content = f.read()

    datasets = content.split("begin dataset")[1:]
    files = {}
    i = 0

    for ds in datasets:
        if "end dataset" not in ds:
            continue
        block = ds.split("end dataset")[0]
        i += 1
        includes = re.findall(r'include\s+"(.+?)"', block)
        files[i] = includes

    if not isinstance(snapshot, int) or snapshot > i:
        snapshot = i

    data = parse_snapshot(files[snapshot], unknown=unknown, folder=folder)
    return process_coords(data, unknown=unknown)

def process_coords(data, unknown=None):
    sorted_data = sorted(data.items(), key = lambda x: (x[0][1], x[0][0]))
    coords = [item[0] for item in sorted_data]
    if len(coords) == 81:
        # might need to change how this works depending on how i get my data
        grid = np.array(coords).reshape((9, 9, 2))
        selected_coords = [tuple(grid[i, j]) for i in [1, 4, 7] for j in [1, 4, 7]]
        filtered_items = [(k, v) for k, v in sorted_data if k in selected_coords]
    elif len(coords) == 9:
        filtered_items = sorted_data
    else:
        raise ValueError("data does not have either 9 or 81 patches.")
    processed = [vals[key] for _, vals in filtered_items for key in unknown]
    return np.concatenate(processed)

def forward_model(parameters, config, model, unknown='h1u', original_call=True, folder="sim_files"):
    if unknown not in variables:
        raise ValueError(f"Unknown variable '{unknown}' is not supported. Choose from {variables}.")
    if isinstance(parameters, Number):
        parameters = [parameters]
    if original_call:
        model.original_call([parameters], config)
    else:
        model([parameters], config)
    return postprocessing("./" + folder + "/solutions/solution-LandslideTsunamiSolver.peano-patch-file",
                          snapshot='last', unknown=[unknown], folder=folder)

def autocorrelation(data, lag):
    """
    Calculate autocorrelation of data at lag value.
    """
    if not lag:
        return 1
    elif lag < 0 or lag % 1:
        raise ValueError("lag must be a non-negative integer.")
    x_t = data[:-lag]
    x_tn = data[lag:]
    cov = np.cov(x_t, x_tn)[0]
    return cov[1]/cov[0]

def autocorrelation_data(data, total_lag):
    """
    Calculate autocorrelation data from lag=0 to lag=total_lag.
    """
    ac_values = [1]
    for i in range(1, total_lag+1):
        ac_values.append(autocorrelation(data, i))
    return ac_values

def ess(data, total_lag):
    data = np.asarray(data, dtype=float)
    n = data.size
    if n < 3:
        return float(n)
    if total_lag is None:
        total_lag = n - 2
    s = 0.0
    for k in range(1, total_lag + 1):
        rk = autocorrelation(data, k)
        if not np.isfinite(rk) or rk <= 0:
            break
        s += rk
    tau = 1 + 2*s
    return n / tau