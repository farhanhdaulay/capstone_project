#!/usr/bin/env python3
# Copyright (c) 2026 Kishore Sridhar - 611451003
# Tatung University — I4210 AI實務專題

import re
import csv
import pandas as pd

LOG_FILE = "/home/jetson/tegrastats.log"
CSV_FILE = "utilization.csv"

def parse_tegrastats():
    # Regex patterns for extracting tegrastats data
    time_re = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")
    ram_re = re.compile(r"RAM (\d+)/\d+MB")
    cpu_re = re.compile(r"CPU \[([^\]]+)\]")
    gpu_re = re.compile(r"GR3D_FREQ (\d+)%")
    cpu_temp_re = re.compile(r"cpu@([\d\.]+)C")
    gpu_temp_re = re.compile(r"gpu@([\d\.]+)C")
    vdd_in_re = re.compile(r"VDD_IN (\d+)mW")
    vdd_cv_re = re.compile(r"VDD_CPU_GPU_CV (\d+)mW")
    vdd_soc_re = re.compile(r"VDD_SOC (\d+)mW")

    data_rows = []

    print(f"Parsing {LOG_FILE}...")
    with open(LOG_FILE, "r") as f:
        for line in f:
            # Skip empty lines or malformed lines caused by log interruption
            if not line.strip() or "RAM" not in line:
                continue

            try:
                # Time
                t_match = time_re.search(line)
                t = t_match.group(1) if t_match else "Unknown"

                # RAM
                ram = int(ram_re.search(line).group(1))

                # CPU Average
                cpu_str = cpu_re.search(line).group(1)
                cpu_cores = [int(core.split('%')[0]) for core in cpu_str.split(',')]
                cpu_avg_pct = round(sum(cpu_cores) / len(cpu_cores), 2)

                # GPU
                gpu_match = gpu_re.search(line)
                gpu_pct = int(gpu_match.group(1)) if gpu_match else 0

                # Temperatures
                cpu_temp = float(cpu_temp_re.search(line).group(1))
                gpu_temp = float(gpu_temp_re.search(line).group(1))

                # Power (mW)
                vdd_in = int(vdd_in_re.search(line).group(1))
                vdd_soc = int(vdd_soc_re.search(line).group(1))
                vdd_cv_match = vdd_cv_re.search(line)
                
                # Orin Nano uses a combined CPU/GPU rail
                vdd_cpu = int(vdd_cv_match.group(1)) if vdd_cv_match else 0
                vdd_gpu = 0 

                data_rows.append({
                    "t": t,
                    "cpu_avg_pct": cpu_avg_pct,
                    "gpu_pct": gpu_pct,
                    "ram_used_mb": ram,
                    "vdd_in_mw": vdd_in,
                    "vdd_cpu_mw": vdd_cpu,
                    "vdd_gpu_mw": vdd_gpu,
                    "vdd_soc_mw": vdd_soc,
                    "gpu_temp_c": gpu_temp,
                    "cpu_temp_c": cpu_temp
                })
            except AttributeError:
                # Gracefully skip lines that are missing sensors (e.g., interrupted writes)
                continue

    # Write to CSV
    headers = ["t", "cpu_avg_pct", "gpu_pct", "ram_used_mb", "vdd_in_mw", "vdd_cpu_mw", "vdd_gpu_mw", "vdd_soc_mw", "gpu_temp_c", "cpu_temp_c"]
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data_rows)
    
    print(f"Successfully generated {CSV_FILE}")
    return data_rows

def print_statistics():
    df = pd.read_csv(CSV_FILE)
    
    # We don't need stats for the timestamp column
    stats_df = df.drop(columns=['t'])
    
    summary = pd.DataFrame({
        'Mean': stats_df.mean(),
        'p50 (Median)': stats_df.median(),
        'p95': stats_df.quantile(0.95),
        'Max': stats_df.max()
    }).round(2)
    
    print("\n" + "="*50)
    print(" HARDWARE UTILIZATION REPORT TABLES")
    print("="*50)
    print(summary.to_string())
    print("="*50 + "\n")

if __name__ == "__main__":
    parse_tegrastats()
    print_statistics()
