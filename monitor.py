#!/usr/bin/env python3
import os
import time
import argparse
import csv
import json
import subprocess
import sys
import glob
import warnings
from datetime import datetime
from typing import List, Dict, Optional, Any

import psutil
from loguru import logger

# Suppress pynvml deprecation warning
warnings.filterwarnings('ignore', category=FutureWarning, module='pynvml')

try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage,
        nvmlDeviceGetName,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo,
        nvmlSystemGetDriverVersion,
        nvmlDeviceGetGraphicsRunningProcesses_v3,
    )
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

# Jetson power monitor paths
JETSON_INA3221_PATH = "/sys/bus/i2c/drivers/ina3221/"
JETSON_HWMON_PATH = "/sys/class/hwmon/"
JETSON_GPU_PATH = "/sys/devices/platform/bus@0/17000000.gpu"


class JetsonPowerRail:
    """Jetson power rail monitor using INA3221 via hwmon"""

    def __init__(self, hwmon_path: str, channel: int, name: str):
        self.hwmon_path = hwmon_path
        self.channel = channel
        self.name = name
        self.voltage_file = os.path.join(hwmon_path, f"in{channel}_input")
        self.current_file = os.path.join(hwmon_path, f"curr{channel}_input")

    def read_power_w(self) -> Optional[float]:
        """Read instantaneous power in watts (calculated from V*I)"""
        try:
            # Read voltage (in mV) and current (in mA)
            with open(self.voltage_file, "r") as f:
                voltage_mv = float(f.read().strip())
            with open(self.current_file, "r") as f:
                current_ma = float(f.read().strip())
            
            # Calculate power: (mV * mA) / 1000000 = W
            power_w = (voltage_mv * current_ma) / 1_000_000.0
            return power_w
        except (PermissionError, FileNotFoundError, ValueError, IOError):
            return None

    def read_voltage_v(self) -> Optional[float]:
        """Read voltage in volts"""
        try:
            with open(self.voltage_file, "r") as f:
                return float(f.read().strip()) / 1000.0  # mV to V
        except (PermissionError, FileNotFoundError, ValueError, IOError):
            return None

    def read_current_a(self) -> Optional[float]:
        """Read current in amps"""
        try:
            with open(self.current_file, "r") as f:
                return float(f.read().strip()) / 1000.0  # mA to A
        except (PermissionError, FileNotFoundError, ValueError, IOError):
            return None


def discover_jetson_power_rails() -> List[JetsonPowerRail]:
    """Discover Jetson power rails from INA3221 sensors via hwmon"""
    rails: List[JetsonPowerRail] = []

    # Method 1: Search via INA3221 driver path
    if os.path.isdir(JETSON_INA3221_PATH):
        device_dirs = glob.glob(os.path.join(JETSON_INA3221_PATH, "*-*"))
        for device_dir in device_dirs:
            if not os.path.isdir(device_dir):
                continue
            # Find hwmon subdirectory
            hwmon_dirs = glob.glob(os.path.join(device_dir, "hwmon/hwmon*"))
            for hwmon_dir in hwmon_dirs:
                rails.extend(_discover_rails_from_hwmon(hwmon_dir))
    
    # Method 2: Search via hwmon class path
    if not rails and os.path.isdir(JETSON_HWMON_PATH):
        hwmon_dirs = glob.glob(os.path.join(JETSON_HWMON_PATH, "hwmon*"))
        for hwmon_dir in hwmon_dirs:
            # Check if this is an INA3221 device
            name_file = os.path.join(hwmon_dir, "name")
            if os.path.isfile(name_file):
                try:
                    with open(name_file, "r") as f:
                        name = f.read().strip()
                    if "ina3221" in name.lower():
                        rails.extend(_discover_rails_from_hwmon(hwmon_dir))
                except Exception:
                    pass

    return rails


def _discover_rails_from_hwmon(hwmon_dir: str) -> List[JetsonPowerRail]:
    """Discover power rails from a hwmon directory"""
    rails: List[JetsonPowerRail] = []
    
    # Find all label files (in1_label, in2_label, etc.)
    label_files = glob.glob(os.path.join(hwmon_dir, "in*_label"))
    
    for label_file in label_files:
        try:
            # Extract channel number from filename (e.g., "in1_label" -> 1)
            basename = os.path.basename(label_file)
            channel = int(basename.replace("in", "").replace("_label", ""))
            
            # Read rail name
            with open(label_file, "r") as f:
                rail_name = f.read().strip()
            
            # Check if corresponding current and voltage files exist
            voltage_file = os.path.join(hwmon_dir, f"in{channel}_input")
            current_file = os.path.join(hwmon_dir, f"curr{channel}_input")
            
            if os.path.isfile(voltage_file) and os.path.isfile(current_file):
                rail = JetsonPowerRail(hwmon_dir, channel, rail_name)
                # Test read
                if rail.read_power_w() is not None:
                    rails.append(rail)
                    logger.debug(f"Found power rail: {rail_name} (channel {channel}) at {hwmon_dir}")
        except (ValueError, IOError) as e:
            logger.debug(f"Error processing {label_file}: {e}")
    
    return rails


class JetsonGpuStats:
    """Read Jetson GPU statistics from sysfs"""
    
    def __init__(self, gpu_path: str = JETSON_GPU_PATH):
        self.gpu_path = gpu_path
        self.load_file = os.path.join(gpu_path, "load")
        self.freq_file = os.path.join(gpu_path, "devfreq/17000000.gpu/cur_freq")
    
    def read_load(self) -> Optional[int]:
        """Read GPU load (0-1000, representing 0-100%)"""
        try:
            with open(self.load_file, "r") as f:
                return int(f.read().strip())
        except (PermissionError, FileNotFoundError, ValueError, IOError):
            return None
    
    def read_freq_mhz(self) -> Optional[float]:
        """Read current GPU frequency in MHz"""
        try:
            with open(self.freq_file, "r") as f:
                freq_hz = int(f.read().strip())
                return freq_hz / 1_000_000.0
        except (PermissionError, FileNotFoundError, ValueError, IOError):
            return None


class GpuDevice:
    def __init__(self, index: int, handle: Any, name: str):
        self.index = index
        self.handle = handle
        self.name = name


def init_nvml_devices() -> List[GpuDevice]:
    devices: List[GpuDevice] = []
    if not NVML_AVAILABLE:
        return devices
    try:
        nvmlInit()
        count = nvmlDeviceGetCount()
        for i in range(count):
            h = nvmlDeviceGetHandleByIndex(i)
            name = (
                nvmlDeviceGetName(h).decode("utf-8")
                if isinstance(nvmlDeviceGetName(h), bytes)
                else nvmlDeviceGetName(h)
            )
            devices.append(GpuDevice(i, h, name))
    except Exception:
        return []
    return devices


def sample_gpu_power(dev: GpuDevice) -> Optional[float]:
    try:
        return nvmlDeviceGetPowerUsage(dev.handle) / 1000.0
    except Exception:
        return None


def sample_gpu_util(dev: Optional[GpuDevice], jetson_gpu: Optional[JetsonGpuStats] = None) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    
    # Try NVML first (for discrete GPUs)
    if dev:
        try:
            util = nvmlDeviceGetUtilizationRates(dev.handle)
            mem = nvmlDeviceGetMemoryInfo(dev.handle)
            data["gpu_util_percent"] = util.gpu
            data["mem_util_percent"] = util.memory
            data["mem_used_mb"] = mem.used / (1024 * 1024)
            data["mem_total_mb"] = mem.total / (1024 * 1024)
            return data
        except Exception:
            pass
    
    # Fallback to Jetson sysfs (for Tegra integrated GPUs)
    if jetson_gpu:
        load = jetson_gpu.read_load()
        if load is not None:
            # load is in range 0-1000, convert to 0-100%
            data["gpu_util_percent"] = round(load / 10.0, 1)
        
        freq = jetson_gpu.read_freq_mhz()
        if freq is not None:
            data["gpu_freq_mhz"] = round(freq, 0)
    
    return data


def sample_gpu_process_util(
    dev: GpuDevice, target_pid: int
) -> Optional[Dict[str, Any]]:
    try:
        procs = nvmlDeviceGetGraphicsRunningProcesses_v3(dev.handle)
        for p in procs:
            if p.pid == target_pid:
                return {"gpu_proc_mem_used_mb": p.usedGpuMemory / (1024 * 1024)}
    except Exception:
        return None
    return None


def run_command(command: List[str]) -> subprocess.Popen:
    return subprocess.Popen(command)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Jetson Power Monitor (INA3221 + GPU)"
    )
    p.add_argument(
        "--interval", type=float, default=1.0, help="Sampling interval in seconds"
    )
    p.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Total monitoring duration in seconds (0=infinite)",
    )
    p.add_argument(
        "--output", type=str, default="", help="Output file path (empty for stdout)"
    )
    p.add_argument(
        "--format", choices=["csv", "jsonl"], default="csv", help="Output format"
    )
    p.add_argument(
        "command",
        type=str,
        nargs=argparse.REMAINDER,
        help='External command to run while monitoring (specify after "--")',
    )
    p.add_argument(
        "--pid",
        type=int,
        default=0,
        help="Monitor existing process PID (when command is not specified)",
    )
    p.add_argument("--no-gpu", action="store_true", help="Disable GPU monitoring")
    p.add_argument(
        "--no-power", action="store_true", help="Disable power rail monitoring"
    )
    p.add_argument("--no-cpu", action="store_true", help="Disable CPU monitoring")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Logger setup
    logger.remove()
    logger.add(sys.stderr, level=args.log_level, enqueue=True, colorize=True)
    logger.info("Jetson Power Monitor - interval={:.3f}s format={}", args.interval, args.format)

    # Discover power rails
    power_rails = discover_jetson_power_rails() if not args.no_power else []
    
    # Initialize GPU devices
    gpu_devices = init_nvml_devices() if (not args.no_gpu and NVML_AVAILABLE) else []
    
    # Initialize Jetson GPU stats reader
    jetson_gpu = None
    if not args.no_gpu and os.path.isdir(JETSON_GPU_PATH):
        jetson_gpu = JetsonGpuStats()
        # Test if we can read GPU stats
        if jetson_gpu.read_load() is None:
            logger.debug("Jetson GPU stats not accessible")
            jetson_gpu = None

    target_process: Optional[psutil.Process] = None
    popen: Optional[subprocess.Popen] = None
    if args.command:
        # Strip leading -- if present in REMAINDER
        cmd = args.command
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        if not cmd:
            logger.error("External command is not specified")
            return
        popen = run_command(cmd)
        target_process = psutil.Process(popen.pid)
        logger.info(f"Monitoring command: {' '.join(cmd)} (PID: {popen.pid})")
    elif args.pid:
        try:
            target_process = psutil.Process(args.pid)
            logger.info(f"Monitoring existing process PID: {args.pid}")
        except Exception:
            logger.error("Cannot access PID {}", args.pid)

    # Report discovered power rails
    if power_rails:
        logger.info(
            "Power rails: {}", ", ".join(f"{r.name}" for r in power_rails)
        )
    else:
        logger.warning("No accessible power rails found")

    # Report GPU devices
    if gpu_devices:
        try:
            drv = nvmlSystemGetDriverVersion().decode()
        except Exception:
            drv = "unknown"
        logger.info(
            "NVIDIA GPUs: {} (driver {})",
            ", ".join(f"{g.index}:{g.name}" for g in gpu_devices),
            drv,
        )
    elif not args.no_gpu and not NVML_AVAILABLE:
        logger.warning("pynvml is not available, GPU monitoring disabled")

    start_time = time.time()
    csv_writer = None
    file_handle = None
    keys_order: List[str] = []

    if args.output:
        mode = "w"
        file_handle = open(args.output, mode, buffering=1)
        if args.format == "csv":
            csv_writer = csv.writer(file_handle)

    header_printed = False

    try:
        while True:
            ts = time.time()
            elapsed = ts - start_time
            if args.duration and elapsed >= args.duration:
                break

            row: Dict[str, Any] = {
                "timestamp": ts,
                "datetime": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[
                    :-3
                ],
            }

            # Sample power rails
            total_power_w = 0.0
            for rail in power_rails:
                power_w = rail.read_power_w()
                if power_w is not None:
                    row[f"power_{rail.name}_w"] = round(power_w, 3)
                    total_power_w += power_w
            
            # Add total power if multiple rails exist
            if len(power_rails) > 1 and total_power_w > 0:
                row["power_total_w"] = round(total_power_w, 3)

            # Sample system CPU usage
            if not args.no_cpu:
                cpu_percent = psutil.cpu_percent(interval=None)
                row["system_cpu_percent"] = round(cpu_percent, 1)
                
                # Sample per-core CPU usage
                cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
                for i, core_pct in enumerate(cpu_per_core):
                    row[f"cpu{i}_percent"] = round(core_pct, 1)

            # Sample GPU power and utilization
            for g in gpu_devices:
                gpw = sample_gpu_power(g)
                if gpw is not None:
                    row[f"gpu{g.index}_power_w"] = round(gpw, 1)
                util = sample_gpu_util(g, jetson_gpu)
                for k, v in util.items():
                    row[f"gpu{g.index}_{k}"] = v
                if target_process:
                    proc_util = sample_gpu_process_util(g, target_process.pid)
                    if proc_util:
                        for k, v in proc_util.items():
                            row[f"gpu{g.index}_{k}"] = v
            
            # If no NVML GPU but Jetson GPU stats available, add them
            if not gpu_devices and jetson_gpu:
                util = sample_gpu_util(None, jetson_gpu)
                for k, v in util.items():
                    row[f"gpu0_{k}"] = v

            # Sample process stats
            if target_process:
                try:
                    cpu_pct = target_process.cpu_percent(interval=None)
                    mem_info = target_process.memory_info()
                    row["proc_cpu_percent"] = cpu_pct
                    row["proc_mem_rss_mb"] = mem_info.rss / (1024 * 1024)
                except Exception:
                    row["proc_ended"] = True
                    target_process = None

            # Output row
            if args.format == "csv":
                if not header_printed:
                    keys_order = list(row.keys())
                    if csv_writer:
                        csv_writer.writerow(keys_order)
                    else:
                        print("#" + ",".join(keys_order))
                    header_printed = True
                line_values = [row.get(k, "") for k in keys_order]
                line = ",".join(str(v) for v in line_values)
                if file_handle:
                    file_handle.write(line + "\n")
                else:
                    print(line)
            else:  # jsonl
                if file_handle:
                    file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    print(json.dumps(row, ensure_ascii=False))

            # Check if command finished
            if popen and popen.poll() is not None and not target_process:
                logger.info("Monitored command finished")
                break

            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if file_handle:
            file_handle.close()
        if NVML_AVAILABLE:
            try:
                nvmlShutdown()
            except Exception:
                pass
        logger.info("Monitoring stopped")


if __name__ == "__main__":
    main()
