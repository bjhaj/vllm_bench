"""
GPU telemetry monitoring for benchmark runs.

Uses NVML (NVIDIA Management Library) to sample GPU metrics during benchmarks:
- GPU utilization (%)
- VRAM usage (MB)
- Temperature (°C)
- Power usage (W)

Samples at ~500ms intervals and writes to CSV.
"""

import asyncio
import csv
import time
import warnings
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None


@dataclass
class GPUMetrics:
    """GPU metrics snapshot."""
    timestamp: str
    elapsed_ms: float
    gpu_id: int
    gpu_utilization_pct: float
    vram_used_mb: float
    vram_total_mb: float
    vram_utilization_pct: float
    temperature_c: Optional[float]
    power_usage_w: Optional[float]


class GPUTelemetryMonitor:
    """
    Monitors GPU metrics during benchmark runs.
    
    Samples GPU stats at regular intervals and records to CSV.
    """
    
    def __init__(
        self,
        output_dir: Path,
        sample_interval_ms: float = 500.0,
        gpu_ids: Optional[List[int]] = None
    ):
        """
        Initialize GPU telemetry monitor.
        
        Args:
            output_dir: Directory to save telemetry CSVs
            sample_interval_ms: Sampling interval in milliseconds
            gpu_ids: List of GPU IDs to monitor (None = all GPUs)
        """
        self.output_dir = Path(output_dir)
        self.sample_interval_ms = sample_interval_ms
        self.gpu_ids = gpu_ids
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.metrics: List[GPUMetrics] = []
        self.start_time: Optional[float] = None
        self.nvml_initialized = False
        
        # Check NVML availability
        if not NVML_AVAILABLE:
            warnings.warn(
                "pynvml not available. GPU telemetry disabled. "
                "Install with: pip install nvidia-ml-py3",
                UserWarning
            )
            return
        
        # Try to initialize NVML
        try:
            pynvml.nvmlInit()
            self.nvml_initialized = True
            
            # Determine which GPUs to monitor
            if self.gpu_ids is None:
                device_count = pynvml.nvmlDeviceGetCount()
                self.gpu_ids = list(range(device_count))
            
            # Validate GPU IDs
            for gpu_id in self.gpu_ids:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    name = pynvml.nvmlDeviceGetName(handle)
                    print(f"[Telemetry] GPU {gpu_id}: {name}")
                except pynvml.NVMLError as e:
                    warnings.warn(f"Cannot access GPU {gpu_id}: {e}", UserWarning)
        
        except pynvml.NVMLError as e:
            warnings.warn(
                f"Failed to initialize NVML: {e}. GPU telemetry disabled.",
                UserWarning
            )
            self.nvml_initialized = False
    
    def _sample_gpu(self, gpu_id: int, elapsed_ms: float) -> Optional[GPUMetrics]:
        """
        Sample metrics for a single GPU.
        
        Args:
            gpu_id: GPU device ID
            elapsed_ms: Milliseconds since monitoring started
            
        Returns:
            GPUMetrics object or None if sampling fails
        """
        if not self.nvml_initialized:
            return None
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            
            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used_mb = mem_info.used / (1024 ** 2)
            vram_total_mb = mem_info.total / (1024 ** 2)
            vram_util_pct = (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0
            
            # Temperature (optional)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                temp = None
            
            # Power usage (optional)
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            except pynvml.NVMLError:
                power = None
            
            return GPUMetrics(
                timestamp=datetime.utcnow().isoformat(),
                elapsed_ms=elapsed_ms,
                gpu_id=gpu_id,
                gpu_utilization_pct=gpu_util,
                vram_used_mb=vram_used_mb,
                vram_total_mb=vram_total_mb,
                vram_utilization_pct=vram_util_pct,
                temperature_c=temp,
                power_usage_w=power
            )
        
        except pynvml.NVMLError as e:
            # Don't spam warnings - just skip this sample
            return None
    
    async def _monitor_loop(self):
        """Main monitoring loop that samples GPU metrics."""
        self.start_time = time.perf_counter()
        
        while self.running:
            current_time = time.perf_counter()
            elapsed_ms = (current_time - self.start_time) * 1000
            
            # Sample all GPUs
            for gpu_id in self.gpu_ids:
                metrics = self._sample_gpu(gpu_id, elapsed_ms)
                if metrics:
                    self.metrics.append(metrics)
            
            # Sleep for sample interval
            await asyncio.sleep(self.sample_interval_ms / 1000.0)
    
    async def start(self):
        """Start monitoring GPU metrics."""
        if not self.nvml_initialized:
            return
        
        if self.running:
            warnings.warn("Telemetry already running", UserWarning)
            return
        
        self.running = True
        self.metrics = []
        self.task = asyncio.create_task(self._monitor_loop())
        print(f"[Telemetry] Started monitoring GPUs {self.gpu_ids} at {self.sample_interval_ms}ms intervals")
    
    async def stop(self):
        """Stop monitoring GPU metrics."""
        if not self.running:
            return
        
        self.running = False
        
        if self.task:
            # Wait for task to complete
            try:
                await asyncio.wait_for(self.task, timeout=2.0)
            except asyncio.TimeoutError:
                self.task.cancel()
        
        print(f"[Telemetry] Stopped monitoring. Collected {len(self.metrics)} samples")
    
    def save_csv(self, run_id: str):
        """
        Save collected metrics to CSV.
        
        Args:
            run_id: Run identifier for filename
        """
        if not self.metrics:
            print("[Telemetry] No metrics collected, skipping CSV save")
            return
        
        output_path = self.output_dir / f"{run_id}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = [
            'timestamp', 'elapsed_ms', 'gpu_id',
            'gpu_utilization_pct', 'vram_used_mb', 'vram_total_mb', 'vram_utilization_pct',
            'temperature_c', 'power_usage_w'
        ]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for metric in self.metrics:
                writer.writerow(asdict(metric))
        
        print(f"[Telemetry] Saved {len(self.metrics)} samples to {output_path}")
    
    def clear(self):
        """Clear collected metrics."""
        self.metrics = []
    
    def shutdown(self):
        """Shutdown NVML."""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self.nvml_initialized = False
            except pynvml.NVMLError:
                pass


async def main():
    """Test/demo for telemetry monitoring."""
    print("GPU Telemetry Test")
    print("=" * 60)
    
    if not NVML_AVAILABLE:
        print("ERROR: pynvml not installed")
        print("Install with: pip install nvidia-ml-py3")
        return
    
    # Create monitor
    monitor = GPUTelemetryMonitor(
        output_dir=Path("results/telemetry"),
        sample_interval_ms=500
    )
    
    if not monitor.nvml_initialized:
        print("ERROR: Failed to initialize NVML")
        return
    
    # Start monitoring
    await monitor.start()
    
    # Simulate some work
    print("\nSimulating 5 seconds of work...")
    await asyncio.sleep(5.0)
    
    # Stop monitoring
    await monitor.stop()
    
    # Save results
    run_id = f"test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    monitor.save_csv(run_id)
    
    # Print summary
    if monitor.metrics:
        print("\nSummary:")
        for gpu_id in monitor.gpu_ids:
            gpu_metrics = [m for m in monitor.metrics if m.gpu_id == gpu_id]
            if gpu_metrics:
                avg_util = sum(m.gpu_utilization_pct for m in gpu_metrics) / len(gpu_metrics)
                avg_vram = sum(m.vram_used_mb for m in gpu_metrics) / len(gpu_metrics)
                print(f"  GPU {gpu_id}: {avg_util:.1f}% avg util, {avg_vram:.0f}MB avg VRAM")
    
    # Cleanup
    monitor.shutdown()
    print("\nTest complete!")


if __name__ == "__main__":
    asyncio.run(main())
