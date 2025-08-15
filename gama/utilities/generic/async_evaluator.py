import contextlib
import datetime
import gc
import logging
import multiprocessing
import os
import psutil
import queue
import struct
import time
import traceback
import uuid
from typing import Optional, Callable, Dict, List
from psutil import NoSuchProcess

log = logging.getLogger(__name__)


class AsyncFuture:
    """Reference to a function call executed on a different process."""

    def __init__(self, fn, *args, **kwargs):
        self.id = uuid.uuid4()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.exception = None
        self.traceback = None

    def execute(self, extra_kwargs):
        try:
            kwargs = {**self.kwargs, **extra_kwargs}
            self.result = self.fn(*self.args, **kwargs)
        except Exception as e:
            self.exception = e
            self.traceback = traceback.format_exc()


class AsyncEvaluator:
    """Manages subprocesses on which arbitrary functions can be evaluated."""

    defaults: Dict = {}

    def __init__(
        self,
        n_workers: int = 1,
        memory_limit_mb: Optional[int] = None,
        logfile: Optional[str] = None,
        wait_time_before_forced_shutdown: int = 10,
        max_runtime_s: Optional[int] = None,
    ):
        self._has_entered = False
        self.futures: Dict[uuid.UUID, AsyncFuture] = {}
        self._processes: List[psutil.Process] = []
        self._n_jobs = n_workers
        self._memory_limit_mb = memory_limit_mb
        self._mem_violations = 0
        self._mem_behaved = 0
        self._logfile = logfile
        self._wait_time_before_forced_shutdown = wait_time_before_forced_shutdown
        self.max_runtime_s = max_runtime_s
        self._running_by_pid = {}

        self.job_queue_size = -n_workers
        self._input: multiprocessing.Queue = multiprocessing.Queue()
        self._output: multiprocessing.Queue = multiprocessing.Queue()
        self._command: multiprocessing.Queue = multiprocessing.Queue()
        self._main_process = psutil.Process(os.getpid())

    def __enter__(self):
        if self._has_entered:
            raise RuntimeError(
                "You can not use the same AsyncEvaluator in two different contexts."
            )
        self._has_entered = True

        self._input = multiprocessing.Queue()
        self._output = multiprocessing.Queue()
        self._command = multiprocessing.Queue()

        log.debug(
            f"Process {self._main_process.pid} starting {self._n_jobs} subprocesses."
        )
        for _ in range(self._n_jobs):
            self._start_worker_process()
        self._log_memory_usage()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        log.debug(f"Signaling {len(self._processes)} subprocesses to stop.")
        # Ask workers to stop gracefully
        for _ in self._processes:
            with contextlib.suppress(Exception):
                self._command.put("stop")

        # Give them a moment
        for _ in range(self._wait_time_before_forced_shutdown + 1):
            # if all workers already exited, break early
            if all(not p.is_running() for p in self._processes):
                break
            time.sleep(1)

        # Hard-stop any stragglers (do this BEFORE closing queues)
        while len(self._processes) > 0:
            try:
                self._stop_worker_process(self._processes[0], force=True)
            except psutil.NoSuchProcess:
                self.job_queue_size -= 1
                self._processes.pop(0)

        # Now it’s safe to drain/close queues
        self.clear_queue(self._input)
        self.clear_queue(self._output)
        self.clear_queue(self._command)

        self.job_queue_size = -self._n_jobs
        return False

    def clear_queue(self, q: multiprocessing.Queue):
        while not q.empty():
            with contextlib.suppress(queue.Empty):
                q.get(timeout=0.001)
        q.close()
        with contextlib.suppress(Exception):
            q.join_thread()

    def submit(self, fn: Callable, *args, **kwargs) -> AsyncFuture:
        future = AsyncFuture(fn, *args, **kwargs)
        self.futures[future.id] = future
        self._input.put(future)
        self.job_queue_size += 1
        return future

    def wait_next(self, poll_time: float = 0.05) -> AsyncFuture:
        if len(self.futures) == 0:
            raise RuntimeError("No Futures queued, must call `submit` first.")
        while True:
            self._control_limits()
            try:
                item = self._output.get(block=False)
            except queue.Empty:
                time.sleep(poll_time)
                continue

            # Special job start signal
            if isinstance(item, tuple) and item[0] == "__job_start__":
                _, pid, start_time = item
                self._running_by_pid[pid] = start_time
                continue

            completed_future = item
            self.job_queue_size -= 1
            match = self.futures.pop(completed_future.id, None)

            if match is None:
                # orphan or cancelled job — still return to caller
                return completed_future

            match.result, match.exception, match.traceback = (
                completed_future.result,
                completed_future.exception,
                completed_future.traceback,
            )
            self._mem_behaved += 1
            # Clear runtime tracking for finished worker
            self._running_by_pid.pop(
                next((p.pid for p in self._processes if p.pid in self._running_by_pid), None),
                None,
            )
            return match

    def _start_worker_process(self) -> psutil.Process:
        mp_process = multiprocessing.Process(
            target=evaluator_daemon,
            args=(self._input, self._output, self._command, AsyncEvaluator.defaults),
            daemon=True,
        )
        mp_process.start()
        subprocess = psutil.Process(mp_process.pid)
        self._processes.append(subprocess)
        return subprocess

    def _stop_worker_process(self, process: psutil.Process, force=False):
        try:
            if not force:
                try:
                    # Only try if queue not obviously closed
                    if not getattr(self._command, "_closed", False):
                        self._command.put("stop")
                    process.wait(timeout=3)
                except (psutil.TimeoutExpired, ValueError):
                    # ValueError -> queue closed; fall through to kill
                    pass

            if process.is_running():
                process.kill()
                process.wait(timeout=5)
        except psutil.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        except NoSuchProcess:
            pass
        finally:
            self.job_queue_size -= 1
            if process in self._processes:
                self._processes.remove(process)
            self._running_by_pid.pop(process.pid, None)

    def _control_limits(self):
        """Memory + runtime control in one place."""
        self._control_memory_usage()
        self._control_runtime_limit()

    def _control_runtime_limit(self):
        if not self.max_runtime_s:
            return
        now = time.time()
        for pid, start in list(self._running_by_pid.items()):
            if now - start > self.max_runtime_s:
                log.info(f"Worker {pid} exceeded max runtime ({self.max_runtime_s}s). Killing.")
                try:
                    proc = next(p for p in self._processes if p.pid == pid)
                    self._stop_worker_process(proc, force=True)
                    self._start_worker_process()
                except StopIteration:
                    pass
                self._running_by_pid.pop(pid, None)

    def _control_memory_usage(self, threshold=0.05):
        if self._memory_limit_mb is None:
            return
        mem_proc = list(self._get_memory_usage())
        if sum(mem for _, mem in mem_proc) > self._memory_limit_mb:
            log.info(
                f"GAMA exceeded memory usage "
                f"({self._mem_violations}, {self._mem_behaved})."
            )
            self._log_memory_usage()
            self._mem_violations += 1

            proc, _ = max(mem_proc[1:], key=lambda t: t[1])
            n_evaluations = self._mem_violations + self._mem_behaved
            fail_ratio = self._mem_violations / n_evaluations

            log.info(f"Requesting stop of worker {proc.pid} due to memory usage.")
            try:
                self._stop_worker_process(proc, force=True)
                if fail_ratio < threshold or len(self._processes) == 1:
                    self._start_worker_process()
                else:
                    self._mem_behaved = 0
                    self._mem_violations = 0
            except Exception as e:
                log.error(f"Error stopping worker {proc.pid}: {e}")

    def _log_memory_usage(self):
        if not self._logfile:
            return
        mem_by_pid = self._get_memory_usage()
        mem_str = ",".join([f"{proc.pid},{mem_mb}" for (proc, mem_mb) in mem_by_pid])
        timestamp = datetime.datetime.now().isoformat()

        with open(self._logfile, "a") as memory_log:
            memory_log.write(f"{timestamp},{mem_str}\n")

    def _get_memory_usage(self):
        processes = [self._main_process] + self._processes
        for process in processes:
            try:
                yield process, process.memory_info()[0] / (2**20)
            except NoSuchProcess:
                self._processes = [p for p in self._processes if p.pid != process.pid]
                self._start_worker_process()


def evaluator_daemon(
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    command_queue: queue.Queue,
    default_parameters: Optional[Dict] = None,
):
    try:
        while True:
            with contextlib.suppress(queue.Empty):
                if not command_queue.empty():
                    cmd = command_queue.get(block=False)
                    if cmd == "stop":
                        break
            try:
                future = input_queue.get(block=False)
                # Notify parent this worker started work
                output_queue.put(("__job_start__", os.getpid(), time.time()))
                future.execute(default_parameters or {})
                output_queue.put(future)
            except (MemoryError, struct.error) as e:
                future.result = None
                future.exception = str(type(e))
                gc.collect()
                output_queue.put(future)
            except queue.Empty:
                pass
    except Exception as e:
        print(f"Stopping daemon:{type(e)}:{str(e)}")
        traceback.print_exc()
