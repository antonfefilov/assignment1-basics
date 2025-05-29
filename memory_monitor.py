#!/usr/bin/env python3
"""Standalone memory monitoring script for any process"""

import argparse
import time
import psutil
import sys
import signal
from typing import List, Tuple, Optional


class ProcessMemoryMonitor:
    """Monitor memory usage of a specific process"""

    def __init__(self, pid: Optional[int] = None, process_name: Optional[str] = None, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.memory_usage: List[Tuple[float, float, float]] = []  # (timestamp, memory_mb, cpu_percent)
        self.peak_memory = 0.0
        self.process: psutil.Process

        if pid:
            try:
                self.process = psutil.Process(pid)
                print(f"Monitoring process PID {pid}: {self.process.name()}")
            except psutil.NoSuchProcess:
                print(f"Error: No process found with PID {pid}")
                sys.exit(1)
        elif process_name:
            processes = [
                p for p in psutil.process_iter(["pid", "name"]) if process_name.lower() in p.info["name"].lower()
            ]
            if not processes:
                print(f"Error: No process found with name containing '{process_name}'")
                sys.exit(1)
            elif len(processes) > 1:
                print(f"Multiple processes found with name '{process_name}':")
                for p in processes:
                    print(f"  PID {p.info['pid']}: {p.info['name']}")
                print("Please specify a PID instead.")
                sys.exit(1)
            else:
                self.process = psutil.Process(processes[0].info["pid"])
                print(f"Monitoring process PID {self.process.pid}: {self.process.name()}")
        else:
            print("Error: Must specify either --pid or --process-name")
            sys.exit(1)

    def start_monitoring(self, duration: Optional[float] = None):
        """Start monitoring the process"""
        self.monitoring = True
        self.memory_usage = []
        self.peak_memory = 0.0

        print(f"Starting memory monitoring (interval: {self.interval}s)")
        if duration:
            print(f"Will monitor for {duration} seconds")
        print("Press Ctrl+C to stop monitoring\n")

        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

        start_time = time.time()
        end_time = start_time + duration if duration else None

        try:
            while self.monitoring:
                current_time = time.time()

                # Check if duration has elapsed
                if end_time and current_time >= end_time:
                    break

                try:
                    # Check if process still exists
                    if not self.process.is_running():
                        print(f"\nProcess {self.process.pid} has terminated.")
                        break

                    # Get memory and CPU info
                    memory_info = self.process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
                    cpu_percent = self.process.cpu_percent()

                    timestamp = current_time - start_time
                    self.memory_usage.append((timestamp, memory_mb, cpu_percent))

                    # Update peak memory
                    if memory_mb > self.peak_memory:
                        self.peak_memory = memory_mb

                    # Print real-time stats
                    print(
                        f"\rTime: {timestamp:6.1f}s | Memory: {memory_mb:8.2f} MB | Peak: {self.peak_memory:8.2f} MB | CPU: {cpu_percent:5.1f}%",
                        end="",
                        flush=True,
                    )

                    time.sleep(self.interval)

                except psutil.NoSuchProcess:
                    print(f"\nProcess {self.process.pid} no longer exists.")
                    break
                except Exception as e:
                    print(f"\nError monitoring process: {e}")
                    break

        except KeyboardInterrupt:
            pass

        self.monitoring = False
        print("\n\nMonitoring stopped.")
        self.print_summary()

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        self.monitoring = False

    def print_summary(self):
        """Print monitoring summary"""
        if not self.memory_usage:
            print("No data collected.")
            return

        memories = [mem for _, mem, _ in self.memory_usage]
        cpu_values = [cpu for _, _, cpu in self.memory_usage]

        avg_memory = sum(memories) / len(memories)
        min_memory = min(memories)
        max_memory = max(memories)
        avg_cpu = sum(cpu_values) / len(cpu_values)
        max_cpu = max(cpu_values)

        duration = self.memory_usage[-1][0]

        print("\n" + "=" * 60)
        print("PROCESS MONITORING SUMMARY")
        print("=" * 60)
        print(f"Process:              {self.process.name()} (PID: {self.process.pid})")
        print(f"Monitoring Duration:  {duration:.2f} seconds")
        print(f"Samples Collected:    {len(self.memory_usage)}")
        print()
        print("MEMORY STATISTICS:")
        print(f"  Peak Memory:        {max_memory:.2f} MB")
        print(f"  Average Memory:     {avg_memory:.2f} MB")
        print(f"  Minimum Memory:     {min_memory:.2f} MB")
        print(f"  Memory Growth:      {max_memory - min_memory:+.2f} MB")
        print()
        print("CPU STATISTICS:")
        print(f"  Peak CPU Usage:     {max_cpu:.1f}%")
        print(f"  Average CPU Usage:  {avg_cpu:.1f}%")
        print("=" * 60)

    def save_log(self, filename: str):
        """Save monitoring data to CSV file"""
        if not self.memory_usage:
            print("No data to save.")
            return

        with open(filename, "w") as f:
            f.write("timestamp_seconds,memory_mb,cpu_percent\n")
            for timestamp, memory, cpu in self.memory_usage:
                f.write(f"{timestamp:.3f},{memory:.2f},{cpu:.2f}\n")

        print(f"Monitoring data saved to: {filename}")


def list_python_processes():
    """List all running Python processes"""
    python_processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["name"] and "python" in proc.info["name"].lower():
                cmdline = " ".join(proc.info["cmdline"]) if proc.info["cmdline"] else ""
                python_processes.append((proc.info["pid"], proc.info["name"], cmdline))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if python_processes:
        print("Running Python processes:")
        print(f"{'PID':<8} {'Name':<15} {'Command Line'}")
        print("-" * 80)
        for pid, name, cmdline in python_processes:
            cmdline_short = cmdline[:50] + "..." if len(cmdline) > 50 else cmdline
            print(f"{pid:<8} {name:<15} {cmdline_short}")
    else:
        print("No Python processes found.")


def main():
    parser = argparse.ArgumentParser(description="Monitor memory usage of a process")
    parser.add_argument("--pid", type=int, help="Process ID to monitor")
    parser.add_argument("--process-name", type=str, help="Process name to monitor (partial match)")
    parser.add_argument("--interval", type=float, default=1.0, help="Monitoring interval in seconds (default: 1.0)")
    parser.add_argument("--duration", type=float, help="Duration to monitor in seconds (default: unlimited)")
    parser.add_argument("--save-log", type=str, help="Save monitoring data to CSV file")
    parser.add_argument("--list-python", action="store_true", help="List all running Python processes and exit")

    args = parser.parse_args()

    if args.list_python:
        list_python_processes()
        return

    if not args.pid and not args.process_name:
        print("Error: Must specify either --pid or --process-name")
        print("Use --list-python to see running Python processes")
        sys.exit(1)

    # Create and start monitor
    monitor = ProcessMemoryMonitor(args.pid, args.process_name, args.interval)
    monitor.start_monitoring(args.duration)

    # Save log if requested
    if args.save_log:
        monitor.save_log(args.save_log)


if __name__ == "__main__":
    main()
