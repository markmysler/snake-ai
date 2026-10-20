"""Measure this machine (CPU cores, RAM, power) and plan how a run uses it.

    python system.py          # print what was detected

No third-party packages: Windows, Linux and macOS are read with their own system calls.
"""
import os
import platform
import subprocess
import sys

import vision as visions
from algorithms import ALGORITHMS


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def cgroup_cpu_limit():
    """CPUs allowed by a Linux container quota (e.g. docker run --cpus=4), or None if unlimited."""
    v2 = _read("/sys/fs/cgroup/cpu.max")  # "max 100000" or "400000 100000"
    if v2:
        quota, _, period = v2.partition(" ")
        if quota != "max" and period:
            return max(1, -(-int(quota) // int(period)))
        return None
    quota, period = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period and int(quota) > 0:
        return max(1, -(-int(quota) // int(period)))
    return None


def cgroup_memory_mb():
    """(limit, available) MB of a Linux container memory limit (docker run --memory), or (None, None)."""
    limit, used = _read("/sys/fs/cgroup/memory.max"), _read("/sys/fs/cgroup/memory.current")  # cgroup v2
    if limit is None:
        limit, used = (_read("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
                       _read("/sys/fs/cgroup/memory/memory.usage_in_bytes"))  # cgroup v1
    if not limit or limit == "max" or int(limit) >= 2**60:  # v1 reports "no limit" as a huge number
        return None, None
    limit_mb = int(limit) // 2**20
    return limit_mb, max(0, limit_mb - int(used or 0) // 2**20)


def cpu_count():
    """CPUs this process may use: CPU affinity and, in a Linux container, its CPU quota."""
    if hasattr(os, "process_cpu_count"):  # Python 3.13+ (affinity)
        n = os.process_cpu_count() or 1
    elif hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
    else:
        n = os.cpu_count() or 1
    if sys.platform.startswith("linux"):
        n = min(n, cgroup_cpu_limit() or n)
    return n


def memory_mb():
    """(total, available) RAM in MB, or (None, None) if unknown."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MemoryStatus()
            m.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys // 2**20, m.ullAvailPhys // 2**20
        if sys.platform.startswith("linux"):
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    key, value = line.split(":", 1)
                    info[key] = int(value.split()[0]) // 1024  # kB -> MB
            total, available = info["MemTotal"], info.get("MemAvailable", info.get("MemFree"))
            limit, room = cgroup_memory_mb()  # in a container, /proc/meminfo shows the whole host
            if limit is not None:
                total, available = min(total, limit), min(available, room)
            return total, available
        if sys.platform == "darwin":
            total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) // 2**20
            pages = subprocess.check_output(["vm_stat"], text=True)
            free = sum(int(line.split()[-1].rstrip(".")) for line in pages.splitlines()
                       if line.startswith(("Pages free", "Pages inactive", "Pages speculative")))
            return total, free * 4096 // 2**20
    except Exception:
        pass
    return None, None


def on_battery():
    """True / False, or None when unknown. Battery power often slows the CPU down."""
    try:
        if sys.platform == "win32":
            import ctypes

            class PowerStatus(ctypes.Structure):
                _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                            ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                            ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
            ps = PowerStatus()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(ps)) and ps.ACLineStatus in (0, 1):
                return ps.ACLineStatus == 0
        elif sys.platform.startswith("linux"):
            supplies = "/sys/class/power_supply"
            for name in os.listdir(supplies):
                path = os.path.join(supplies, name, "online")
                if name.startswith(("AC", "ADP")) and os.path.exists(path):
                    return open(path).read().strip() == "0"
    except Exception:
        pass
    return None


def detect():
    total, available = memory_mb()
    return {"cpus": cpu_count(), "ram_total_mb": total, "ram_available_mb": available,
            "on_battery": on_battery(), "platform": platform.platform(), "python": platform.python_version()}


def pid_alive(pid):
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


# ---- planning -------------------------------------------------------------------------
MIN_DQN_BUFFER = 10_000


def plan(config, info=None, parallel=1):
    """How a run would use this machine. config = a normalized run config (see runconfig.py).
    `parallel` = how many runs share the machine at once (each gets 1/parallel of it).

    Default split (the same as lessons 01-02): when DQNs run alongside, GA pools get half the
    cores; each DQN gets one thread; the rest is left for the dashboard and the OS."""
    info = info or detect()
    cpus, parallel = info["cpus"], max(1, int(parallel))
    share = max(1, cpus // parallel)
    ram_avail = info["ram_available_mb"] or 4096
    ram_budget = ram_avail * 0.8 / parallel
    n_in = visions.state_size(config["vision"], config["width"], config["height"])
    agents = config["agents"]
    pools = [a for a in agents if ALGORITHMS[a["algorithm"]].resource["role"] == "pool"]
    threaded = [a for a in agents if a not in pools]
    pool_cores = max(1, share // 2 if threaded else share - 1)

    out, warnings = {}, []
    for a in agents:
        algo, params = ALGORITHMS[a["algorithm"]], dict(a["params"])
        if algo.resource["role"] == "pool":
            cores = a.get("cores") or max(1, pool_cores // len(pools))
        else:
            cores = a.get("cores") or 1
        out[a["id"]] = {"algorithm": algo.name, "cores": cores, "flag": algo.resource["flag"],
                        "params": params, "mem_mb": algo.memory_mb(n_in, params, cores)}

    # Too little RAM: shrink replay buffers first, then worker pools.
    def total():
        return sum(p["mem_mb"] for p in out.values())
    for agent_id, p in out.items():
        while total() > ram_budget and "buffer" in p["params"] and p["params"]["buffer"] > MIN_DQN_BUFFER:
            p["params"]["buffer"] = max(MIN_DQN_BUFFER, p["params"]["buffer"] // 2)
            p["mem_mb"] = ALGORITHMS[p["algorithm"]].memory_mb(n_in, p["params"], p["cores"])
            p["shrunk"] = True
        while total() > ram_budget and p["flag"] == "workers" and p["cores"] > 1:
            p["cores"] -= 1
            p["mem_mb"] = ALGORITHMS[p["algorithm"]].memory_mb(n_in, p["params"], p["cores"])
            p["shrunk"] = True
        if p.get("shrunk"):
            warnings.append(f"{agent_id}: reduced to fit in RAM ({p['flag']} {p['cores']}"
                            + (f", buffer {p['params']['buffer']:,}" if "buffer" in p["params"] else "") + ").")
    fits = total() <= ram_budget
    if not fits:
        warnings.append(f"Needs about {total():,.0f} MB but only {ram_budget:,.0f} MB is free for this run.")
    cores_used = sum(p["cores"] for p in out.values())
    if cores_used > share:
        warnings.append(f"Uses {cores_used} cores but {share} are available per run: training will be slower.")
    if info.get("on_battery"):
        warnings.append("Running on battery: the CPU may be throttled, so wall-clock results "
                        "are not comparable with runs made on mains power.")

    # How many copies of this run fit side by side with the minimum resources each.
    min_mem = sum(ALGORITHMS[a["algorithm"]].memory_mb(
        n_in, {**a["params"], **({"buffer": MIN_DQN_BUFFER} if "buffer" in a["params"] else {})}, 1)
        for a in agents)
    min_cores = len(threaded) + (2 * len(pools))
    max_parallel = max(1, min(cpus // max(1, min_cores), int(ram_avail * 0.8 // max(1, min_mem))))
    return {"system": info, "parallel": parallel, "cores_per_run": share, "ram_budget_mb": round(ram_budget),
            "inputs": n_in, "agents": out, "cores_used": cores_used, "mem_mb": round(total()),
            "fits": fits, "warnings": warnings, "max_parallel": max_parallel}


if __name__ == "__main__":
    import json
    print(json.dumps(detect(), indent=2))
