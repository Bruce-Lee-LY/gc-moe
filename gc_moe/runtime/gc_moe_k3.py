"""CUDA driver Green Context resources and the K3 child graph.

The role kernels are CuTe DSL cubins.  A CUDA library kernel plus an explicit
context is required for Graph kernel nodes: ``func`` remains null.
"""

from __future__ import annotations

import atexit
from collections import OrderedDict, deque
import ctypes
from dataclasses import dataclass, field
import errno
from functools import lru_cache, wraps
import hashlib
from itertools import count
import os
import re
import threading
import time
from typing import Sequence

import torch

from ..layout.gc_moe_k3 import (
    COMPUTE_ROLE, DISPATCH_ROLE, FINALIZE_ROLE, FRONT_ROLE, PROJECTION_REDUCE_ROLE,
    ROUTED_ASSIST_ROLE, ROUTER_ROLE, VALID_ROLE_ORDERS,
)
from cuda.bindings import driver as cu


def _ok(result):
    if result[0] != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def _get_env_int(name: str, default: int) -> int:
    """Parse a signed decimal int32 environment value without whitespace."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    if re.fullmatch(r"-?[0-9]+", raw) is None:
        raise ValueError(f"invalid value for {name}: {raw}")
    value = int(raw)
    if not -(1 << 31) <= value < (1 << 31):
        raise ValueError(f"invalid value for {name}: {raw}")
    return value


def _get_requested_dispatch_sms(num_ranks: int,
                                requested_dispatch_sms: int | None = None) -> int:
    """Validate ranks; explicit SM requests bypass environment parsing."""
    if isinstance(num_ranks, bool) or not isinstance(num_ranks, int) or num_ranks < 1:
        raise ValueError('num_ranks must be a positive integer')
    if requested_dispatch_sms is not None:
        return requested_dispatch_sms
    return _get_env_int('GC_MOE_DISPATCH_SMS', 8 if num_ranks <= 8 else 16)


class _GreenResources:
    _UNSUPPORTED_RESOURCE = (cu.CUresult.CUDA_ERROR_INVALID_RESOURCE_TYPE,
                             cu.CUresult.CUDA_ERROR_NOT_SUPPORTED)

    def __init__(self):
        required = (
            'cuDeviceGetDevResource', 'cuDevSmResourceSplitByCount',
            'cuCtxGetDevResource', 'cuDevResourceGenerateDesc',
            'cuGreenCtxCreate', 'cuGreenCtxGetDevResource', 'CUdevResource',
            'CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM',
            'CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG',
            'CUdevWorkqueueConfigScope.CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED',
            'CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM',
        )
        for name in required:
            value = cu
            for component in name.split('.'):
                if not hasattr(value, component):
                    raise RuntimeError(f'cuda.bindings.driver is missing {name}')
                value = getattr(value, component)
        resource = cu.CUdevResource()
        for name in ('sm.smCount', 'sm.minSmPartitionSize',
                     'sm.smCoscheduledAlignment', 'wqConfig.sharingScope',
                     'wqConfig.wqConcurrencyLimit', 'nextResource'):
            value = resource
            for component in name.split('.'):
                if not hasattr(value, component):
                    raise RuntimeError(f'cuda.bindings.CUdevResource is missing {name}')
                value = getattr(value, component)
        self._SM = cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        self._WORKQUEUE_CONFIG = cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG
        self._GREEN_CTX_BALANCED = cu.CUdevWorkqueueConfigScope.CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED

    def get_sm_split(self, device, requested_dispatch_sms):
        whole = _ok(cu.cuDeviceGetDevResource(device, self._SM))
        if whole.type != self._SM:
            raise RuntimeError('CUDA returned a non-SM device resource')
        total = whole.sm.smCount
        alignment = max(2, whole.sm.smCoscheduledAlignment)
        minimum = max(2, whole.sm.minSmPartitionSize)
        if not (minimum <= requested_dispatch_sms < total):
            raise ValueError('invalid dispatch SM request')
        compute_target = ((total - requested_dispatch_sms) // alignment) * alignment
        compute, groups, communication = _ok(
            cu.cuDevSmResourceSplitByCount(1, whole, 0, compute_target))
        if (groups != 1 or len(compute) != 1 or
                compute[0].type != self._SM or communication.type != self._SM):
            raise RuntimeError('CUDA returned an invalid Green SM split')
        compute_sms, dispatch_sms = compute[0].sm.smCount, communication.sm.smCount
        if (compute_sms < 32 or compute_sms % 2 or
                dispatch_sms <= 1 or compute_sms + dispatch_sms != total):
            raise RuntimeError('unsupported Green SM partition')
        return total, compute[0], communication

    def get_balanced_workqueue(self):
        primary = _ok(cu.cuCtxGetCurrent())
        if not int(primary):
            raise RuntimeError('Green workqueue query needs a primary CUDA context')
        result = cu.cuCtxGetDevResource(primary, self._WORKQUEUE_CONFIG)
        if result[0] in self._UNSUPPORTED_RESOURCE:
            return None
        resource = _ok(result)
        if resource.type != self._WORKQUEUE_CONFIG or resource.wqConfig.wqConcurrencyLimit < 2:
            raise RuntimeError('CUDA workqueue resource cannot support two Green workers')
        resource.wqConfig.sharingScope = self._GREEN_CTX_BALANCED
        resource.wqConfig.wqConcurrencyLimit = 1
        return resource

    def make_green(self, device, sm_resource, workqueue):
        resources = [sm_resource] if workqueue is None else [sm_resource, workqueue]
        descriptor = _ok(cu.cuDevResourceGenerateDesc(resources, len(resources)))
        green = _ok(cu.cuGreenCtxCreate(
            descriptor, device, cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
        if not int(green):
            raise RuntimeError('CUDA returned a null Green context')
        # GreenContextPair owns the one matching cuGreenCtxDestroy.
        return green

    def verify_balanced_workqueue(self, green):
        result = cu.cuGreenCtxGetDevResource(green, self._WORKQUEUE_CONFIG)
        if result[0] in self._UNSUPPORTED_RESOURCE:
            return
        actual = _ok(result)
        if (actual.type != self._WORKQUEUE_CONFIG or
                actual.wqConfig.sharingScope != self._GREEN_CTX_BALANCED or
                actual.wqConfig.wqConcurrencyLimit != 1):
            raise RuntimeError('Green context did not retain balanced workqueue limits')


@lru_cache(maxsize=1)
def _green_resources():
    return _GreenResources()


@dataclass(frozen=True)
class RoleBinary:
    role: int
    cubin: bytes
    smem_bytes: int
    threads: int
    grid: int
    cluster: tuple[int, int, int] = (1, 1, 1)
    programmatic_dependencies: tuple[int, ...] = ()
    cubin_digest: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, 'cubin_digest', hashlib.sha256(self.cubin).digest())


class _KernelArgs:
    def __init__(self, values: Sequence[ctypes._SimpleCData]):
        self.values = tuple(values)
        pointers = [ctypes.addressof(value) for value in self.values]
        self.pointers = (ctypes.c_void_p * len(pointers))(*pointers)

    @property
    def pointer(self) -> int:
        return ctypes.addressof(self.pointers)

    @property
    def signature(self) -> bytes:
        return b"".join(ctypes.string_at(ctypes.addressof(x), ctypes.sizeof(x)) for x in self.values)


def _child_key(binaries: Sequence[RoleBinary], arguments: Sequence[_KernelArgs]) -> bytes:
    key = hashlib.sha256()
    for binary, role_args in zip(binaries, arguments):
        key.update(len(role_args.signature).to_bytes(4, 'little'))
        key.update(role_args.signature)
        key.update(binary.cubin_digest)
        key.update(bytes((binary.role,)))
        key.update(binary.smem_bytes.to_bytes(8, 'little'))
        key.update(binary.threads.to_bytes(4, 'little'))
        key.update(binary.grid.to_bytes(4, 'little'))
        key.update(bytes(binary.cluster))
        key.update(bytes((len(binary.programmatic_dependencies), *binary.programmatic_dependencies)))
    return key.digest()


@dataclass
class _LoadedKernel:
    library: object
    kernel: object = None

    def close(self):
        if self.library is not None:
            _ok(cu.cuLibraryUnload(self.library))
            self.library = None
            self.kernel = None


_LIBRARY_LOCK = threading.RLock()
_LOADED_KERNELS: dict[tuple, _LoadedKernel] = {}
_LIVE_PAIRS: dict[int, 'GreenContextPair'] = {}
_RUNTIME_STATE = 'open'
_GRAPH_CACHE_CAPACITY = 16


def _runtime_locked(function):
    @wraps(function)
    def locked(*args, **kwargs):
        with _LIBRARY_LOCK:
            return function(*args, **kwargs)
    return locked


@lru_cache(maxsize=1)
def _native_graph_signals():
    """Native one-way destructor notification; never call Python from CUDA.

    A user-object destructor cannot call CUDA, including a Tensor destructor.
    POSIX sem_post publishes retirement to the calling thread instead. Its
    native return value is ignored by the CUhostFn ABI. libc owns the opaque
    semaphore storage, so no sem_t size/alignment assumptions are required.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    libc.sem_open.argtypes = [ctypes.c_char_p, ctypes.c_int,
                             ctypes.c_uint, ctypes.c_uint]
    libc.sem_open.restype = ctypes.c_void_p
    libc.sem_unlink.argtypes = [ctypes.c_char_p]
    libc.sem_unlink.restype = ctypes.c_int
    for name in ('sem_post', 'sem_trywait', 'sem_close'):
        function = getattr(libc, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_int
    callback = cu.CUhostFn(ctypes.cast(libc.sem_post, ctypes.c_void_p).value)
    return libc, callback


_GRAPH_SIGNAL_IDS = count()
_GRAPH_STORAGE: dict[int, '_GraphStorage'] = {}
_RETIRING_GRAPH_STORAGE: deque[int] = deque()
_FAILED_USER_OBJECTS: list[object] = []


class _GraphStorage:
    """Keep allocations alive until every CUDA graph/execution retires.

    Names are unlinked immediately. Only the native semaphore handle remains;
    if a caller leaves a graph alive at process exit, its native callback can
    still post without entering a finalized Python interpreter.
    """

    def __init__(self, resources: tuple, workspace_ptr: int):
        self.libc, self.callback = _native_graph_signals()
        name = f'/gck3-{os.getpid()}-{next(_GRAPH_SIGNAL_IDS)}'.encode()
        self.signal = self.libc.sem_open(name, os.O_CREAT | os.O_EXCL, 0o600, 0)
        if not self.signal or self.signal == ctypes.c_void_p(-1).value:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        if self.libc.sem_unlink(name):
            code = ctypes.get_errno()
            self.libc.sem_close(self.signal)
            raise OSError(code, os.strerror(code))
        self.resources = resources
        self.workspace_ptr = workspace_ptr

    def retired(self) -> bool:
        if self.libc.sem_trywait(self.signal) == 0:
            return True
        code = ctypes.get_errno()
        if code in (errno.EAGAIN, errno.EINTR):
            return False
        raise OSError(code, os.strerror(code))

    def close(self):
        if self.libc.sem_close(self.signal):
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        self.signal = None
        # Tensor/IPC allocation destructors run on the ordinary API thread.
        self.resources = ()


@_runtime_locked
def _collect_retired_storage(max_checks: int = 8) -> None:
    # Cached graphs necessarily retain their storage. Poll only evicted graph
    # owners, with bounded work even when many outer graphs remain alive.
    for _ in range(min(max_checks, len(_RETIRING_GRAPH_STORAGE))):
        token = _RETIRING_GRAPH_STORAGE.popleft()
        storage = _GRAPH_STORAGE[token]
        if storage.retired():
            storage.close()
            del _GRAPH_STORAGE[token]
        else:
            _RETIRING_GRAPH_STORAGE.append(token)


def _retain_graph_storage(graph, resources: tuple,
                          workspace_ptr: int = 0) -> int | None:
    if not resources:
        return
    storage = _GraphStorage(resources, workspace_ptr)
    token = storage.signal
    _GRAPH_STORAGE[token] = storage
    try:
        user_object = _ok(cu.cuUserObjectCreate(
            token, storage.callback, 1,
            cu.CUuserObject_flags.CU_USER_OBJECT_NO_DESTRUCTOR_SYNC))
    except Exception:
        del _GRAPH_STORAGE[token]
        storage.close()
        raise
    try:
        _ok(cu.cuGraphRetainUserObject(
            graph, user_object, 1,
            cu.CUuserObjectRetain_flags.CU_GRAPH_USER_OBJECT_MOVE))
    except Exception:
        # Failed ownership transfer leaves the initial reference with us.
        # Keep failed releases pinned rather than free potentially live data.
        if cu.cuUserObjectRelease(user_object, 1)[0] != cu.CUresult.CUDA_SUCCESS:
            _FAILED_USER_OBJECTS.append(user_object)
        else:
            _RETIRING_GRAPH_STORAGE.append(token)
        raise
    return token


@_runtime_locked
def _release_workspace_storage(workspace_ptr: int) -> None:
    """Retire cached bindings before the last workspace wrapper detaches."""
    for owner in tuple(_LIVE_PAIRS.values()):
        owner._drain_deferred_graphs()
        for key, graph in tuple(owner.graphs.items()):
            if graph.workspace_ptr == workspace_ptr:
                try:
                    graph.close()
                except Exception:
                    owner._state = 'failed'
                    _pin_failed_pair(owner)
                    raise
                del owner.graphs[key]
    # CUDA destructors notify asynchronously. Captured outer graphs must
    # already be retired by the caller; a violation leaves storage pinned.
    deadline = time.monotonic() + 5
    while any(x.workspace_ptr == workspace_ptr for x in _GRAPH_STORAGE.values()):
        _collect_retired_storage(len(_RETIRING_GRAPH_STORAGE))
        if not any(x.workspace_ptr == workspace_ptr for x in _GRAPH_STORAGE.values()):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('retire captured K3 graphs before destroying their workspace')
        time.sleep(.001)


class _ChildGraph:
    def __init__(self, owner: 'GreenContextPair', binaries: Sequence[RoleBinary],
                 arguments: Sequence[_KernelArgs], keepalive: tuple):
        roles = tuple(x.role for x in binaries)
        if roles not in VALID_ROLE_ORDERS:
            raise ValueError(f'invalid K3 role order {roles}, expected one of {VALID_ROLE_ORDERS}')
        if len(arguments) != len(roles):
            raise ValueError('expected one argument vector per role')
        self.graph = None
        self.executable = None
        self.kernels = []
        self.storage_token = None
        self.arguments = tuple(arguments)
        # Every K3 node ABI starts with the symmetric workspace pointer.
        self.workspace_ptr = int(arguments[0].values[0].value)
        try:
            self.graph = _ok(cu.cuGraphCreate(0))
            nodes = {}
            dep_roles = {FINALIZE_ROLE: (DISPATCH_ROLE, COMPUTE_ROLE)}
            if roles[0] == FRONT_ROLE:
                dep_roles.update({
                    ROUTER_ROLE: (FRONT_ROLE,),
                    DISPATCH_ROLE: (ROUTER_ROLE,),
                    COMPUTE_ROLE: (ROUTER_ROLE if owner.primary_ctx_roles else FRONT_ROLE,),
                })
            if roles[-1] == PROJECTION_REDUCE_ROLE:
                dep_roles[PROJECTION_REDUCE_ROLE] = (FINALIZE_ROLE,)
            for index, binary in enumerate(binaries):
                loaded = owner._get_loaded_kernel(binary)
                kernel = loaded.kernel
                _ok(cu.cuKernelSetAttribute(
                    cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    binary.smem_bytes, kernel, owner.device))
                if binary.role == ROUTED_ASSIST_ROLE:
                    if not owner._can_launch_routed_assist(kernel, binary):
                        continue
                    dep_roles[ROUTED_ASSIST_ROLE] = (DISPATCH_ROLE,)
                    dep_roles[FINALIZE_ROLE] = (COMPUTE_ROLE, ROUTED_ASSIST_ROLE)
                self.kernels.append(loaded)
                params = cu.CUgraphNodeParams()
                params.type = cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
                params.kernel.kern = kernel
                params.kernel.ctx = (owner.primary_context
                                     if owner.primary_ctx_roles else
                                     owner.contexts[0 if binary.role in (DISPATCH_ROLE, ROUTER_ROLE, ROUTED_ASSIST_ROLE) else 1])
                params.kernel.gridDimX = binary.grid
                params.kernel.gridDimY = params.kernel.gridDimZ = 1
                params.kernel.blockDimX = binary.threads
                params.kernel.blockDimY = params.kernel.blockDimZ = 1
                params.kernel.sharedMemBytes = binary.smem_bytes
                params.kernel.kernelParams = arguments[index].pointer
                dependency_roles = dep_roles.get(binary.role, ())
                dependencies = [nodes[role] for role in dependency_roles]
                edge_data = None
                if binary.programmatic_dependencies:
                    edge_data = [cu.CUgraphEdgeData() for _ in dependencies]
                    for role in binary.programmatic_dependencies:
                        edge = edge_data[dependency_roles.index(role)]
                        edge.type = cu.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC
                        edge.from_port = cu.CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC
                node = _ok(cu.cuGraphAddNode(self.graph, dependencies, edge_data,
                                             len(dependencies), params))
                if binary.cluster != (1, 1, 1):
                    attr = cu.CUkernelNodeAttrValue()
                    attr.clusterDim.x, attr.clusterDim.y, attr.clusterDim.z = binary.cluster
                    _ok(cu.cuGraphKernelNodeSetAttribute(
                        node, cu.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, attr))
                nodes[binary.role] = node
            # Child-node clones and instantiated/running graphs retain this
            # user object even after the bounded host cache evicts our graph.
            self.storage_token = _retain_graph_storage(
                self.graph, keepalive, self.workspace_ptr)
            # Capture only needs the template. Instantiate lazily on an eager
            # launch; cuGraphInstantiate is prohibited during stream capture.
        except Exception:
            try:
                self.close()
            except Exception:
                owner._failed_children.append(self)
                owner._state = 'failed'
                _pin_failed_pair(owner)
                raise
            raise

    def instantiate(self):
        if self.executable is None:
            self.executable = _ok(cu.cuGraphInstantiate(self.graph, 0))

    def close(self):
        if self.executable is not None:
            _ok(cu.cuGraphExecDestroy(self.executable))
            self.executable = None
        if self.graph is not None:
            _ok(cu.cuGraphDestroy(self.graph))
            self.graph = None
        if self.storage_token is not None:
            _RETIRING_GRAPH_STORAGE.append(self.storage_token)
            self.storage_token = None
        self.kernels.clear()


class GreenContextPair:
    """CUDA-aligned dispatch/compute partition for the SM100 role graph.

    The default dispatch request is 8 SMs for <=8 ranks, otherwise 16;
    GC_MOE_DISPATCH_SMS overrides it, and an explicit request takes precedence.
    GC_MOE_PRIMARY_CTX=1 binds all role kernels to the primary context and
    uses the split for grid sizes only.
    """

    @_runtime_locked
    def __init__(self, device_index: int | None = None,
                 requested_dispatch_sms: int | None = None, *,
                 num_ranks: int = 1, primary_ctx_roles: bool | None = None):
        if _RUNTIME_STATE != 'open':
            raise RuntimeError(f"kernel runtime is {_RUNTIME_STATE}")
        requested_dispatch_sms = _get_requested_dispatch_sms(
            num_ranks, requested_dispatch_sms)
        torch.cuda.init()
        self.primary_context = _ok(cu.cuCtxGetCurrent())
        self.primary_ctx_roles = (_get_env_int('GC_MOE_PRIMARY_CTX', 0) != 0
                                  if primary_ctx_roles is None else primary_ctx_roles)
        if device_index is None:
            device_index = torch.cuda.current_device()
        self.device = _ok(cu.cuDeviceGet(device_index))
        green_resources = _green_resources()
        _, compute, communication = green_resources.get_sm_split(
            self.device, requested_dispatch_sms)
        workqueue = None
        if not self.primary_ctx_roles and _get_env_int('GC_MOE_BALANCED_WORKQUEUES', 1):
            workqueue = green_resources.get_balanced_workqueue()
        self.compute_sms = compute.sm.smCount
        self.dispatch_sms = communication.sm.smCount
        parts = () if self.primary_ctx_roles else (communication, compute)
        self.balanced_workqueues = workqueue is not None
        self.green = []
        self.contexts = []
        self.streams = []
        self.retirement = []
        self._assist_support: dict[tuple, bool] = {}
        self.graphs: OrderedDict[bytes, _ChildGraph] = OrderedDict()
        self._deferred_graphs: deque[_ChildGraph] = deque()
        self._failed_libraries: list[_LoadedKernel] = []
        self._failed_children: list[_ChildGraph] = []
        self._state = 'open'
        _LIVE_PAIRS[id(self)] = self
        try:
            for part in parts:
                green = green_resources.make_green(self.device, part, workqueue)
                self.green.append(green)
                if workqueue is not None:
                    green_resources.verify_balanced_workqueue(green)
                self.contexts.append(_ok(cu.cuCtxFromGreenCtx(green)))
                self.streams.append(_ok(cu.cuGreenCtxStreamCreate(
                    green, cu.CUstream_flags.CU_STREAM_NON_BLOCKING, 0)))
                actual_green = _ok(cu.cuStreamGetGreenCtx(self.streams[-1]))
                if int(actual_green) != int(green):
                    raise RuntimeError('Green stream belongs to another context')
                self.retirement.append(_ok(cu.cuEventCreate(cu.CUevent_flags.CU_EVENT_DISABLE_TIMING)))
        except Exception:
            try:
                self.close()
            except Exception:
                _pin_failed_pair(self)
                raise
            raise

    @_runtime_locked
    def _get_loaded_kernel(self, binary: RoleBinary) -> _LoadedKernel:
        if _RUNTIME_STATE != 'open':
            raise RuntimeError(f"kernel runtime is {_RUNTIME_STATE}")
        key = (binary.role, binary.cubin_digest)
        loaded = _LOADED_KERNELS.get(key)
        if loaded is not None:
            return loaded
        library = _ok(cu.cuLibraryLoadData(binary.cubin, None, None, 0, None, None, 0))
        loaded = _LoadedKernel(library)
        try:
            if _ok(cu.cuLibraryGetKernelCount(library)) != 1:
                raise RuntimeError('expected exactly one kernel in a role library')
            kernels = _ok(cu.cuLibraryEnumerateKernels(1, library))
            if isinstance(kernels, tuple):
                kernels = kernels[0]
            if len(kernels) != 1:
                raise RuntimeError('role library enumeration changed')
            loaded.kernel = kernels[0]
            _LOADED_KERNELS[key] = loaded
            return loaded
        except Exception:
            try:
                loaded.close()
            except Exception:
                self._failed_libraries.append(loaded)
                self._state = 'failed'
                _pin_failed_pair(self)
                raise
            raise

    def _can_launch_routed_assist(self, kernel, binary: RoleBinary) -> bool:
        # The split remainder has no cluster co-scheduling guarantee. Query the
        # loaded kernel with the communication Green stream and its actual launch.
        key = (int(kernel), binary.grid, binary.threads, binary.smem_bytes, binary.cluster)
        if key not in self._assist_support:
            attr = cu.CUlaunchAttribute()
            attr.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            attr.value.clusterDim.x, attr.value.clusterDim.y, attr.value.clusterDim.z = binary.cluster
            config = cu.CUlaunchConfig()
            config.gridDimX, config.gridDimY, config.gridDimZ = binary.grid, 1, 1
            config.blockDimX, config.blockDimY, config.blockDimZ = binary.threads, 1, 1
            config.sharedMemBytes = binary.smem_bytes
            config.hStream = self.streams[0]
            config.attrs, config.numAttrs = [attr], 1
            result = cu.cuOccupancyMaxActiveClusters(cu.CUfunction(int(kernel)), config)
            if result[0] in (cu.CUresult.CUDA_ERROR_INVALID_CLUSTER_SIZE,
                             cu.CUresult.CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION,
                             cu.CUresult.CUDA_ERROR_NOT_SUPPORTED,
                             cu.CUresult.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
                             cu.CUresult.CUDA_ERROR_INVALID_VALUE):
                supported = False
            else:
                supported = _ok(result) > 0
            self._assist_support[key] = supported
        return self._assist_support[key]

    @_runtime_locked
    def launch(self, binaries: Sequence[RoleBinary], arguments: Sequence,
               stream: int | None = None, *, keepalive: tuple = ()) -> None:
        if _RUNTIME_STATE != 'open':
            raise RuntimeError(f"kernel runtime is {_RUNTIME_STATE}")
        if self._state != 'open':
            raise RuntimeError(f"Green context pair is {self._state}")
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        status, capture_status, _, parent, dependencies, edges, count = cu.cuStreamGetCaptureInfo(stream)
        if status != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cannot inspect CUDA stream capture: {status}")
        if capture_status not in (cu.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE,
                                  cu.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE):
            raise RuntimeError('invalid CUDA stream capture state')
        if len(arguments) != len(binaries):
            raise ValueError('expected one argument vector per K3 node')
        args = tuple(_KernelArgs(item) for item in arguments)
        key = _child_key(binaries, args)
        # Dropping the final Tensor/IPC reference can invoke allocator CUDA
        # calls. Defer that host-side work until outside stream capture.
        if capture_status == cu.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
            self._drain_deferred_graphs()
            _collect_retired_storage()
        graph = self.graphs.get(key)
        if graph is None:
            graph = _ChildGraph(self, binaries, args, keepalive)
            try:
                self.graphs[key] = graph
            except Exception:
                try:
                    graph.close()
                except Exception:
                    self._failed_children.append(graph)
                    self._state = 'failed'
                    _pin_failed_pair(self)
                    raise
                raise
            while len(self.graphs) > _GRAPH_CACHE_CAPACITY:
                old_key = next(iter(self.graphs))
                old_graph = self.graphs[old_key]
                if capture_status == cu.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
                    # Destroying an instantiated executable is also forbidden
                    # by global capture. Drop it from the LRU now and destroy
                    # it after capture, while CUDA keeps the parent clone live.
                    self._deferred_graphs.append(old_graph)
                else:
                    try:
                        old_graph.close()
                    except Exception:
                        self._state = 'failed'
                        _pin_failed_pair(self)
                        raise
                del self.graphs[old_key]
            if capture_status == cu.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
                _collect_retired_storage()
        else:
            self.graphs.move_to_end(key)
        if capture_status == cu.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
            params = cu.CUgraphNodeParams()
            params.type = cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_GRAPH
            params.graph.graph = graph.graph
            child = _ok(cu.cuGraphAddNode(parent, dependencies, edges, count, params))
            _ok(cu.cuStreamUpdateCaptureDependencies(
                stream, [child], None, 1,
                cu.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_SET_CAPTURE_DEPENDENCIES))
        elif capture_status == cu.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
            graph.instantiate()
            _ok(cu.cuGraphLaunch(graph.executable, stream))

    def _drain_deferred_graphs(self) -> None:
        while self._deferred_graphs:
            graph = self._deferred_graphs[0]
            try:
                graph.close()
            except Exception:
                self._state = 'failed'
                _pin_failed_pair(self)
                raise
            self._deferred_graphs.popleft()

    @_runtime_locked
    def _clear_graph_cache(self) -> None:
        """Drop host graph references; CUDA retains in-flight/captured storage."""
        self._drain_deferred_graphs()
        for key, graph in tuple(self.graphs.items()):
            try:
                graph.close()
            except Exception:
                self._state = 'failed'
                _pin_failed_pair(self)
                raise
            del self.graphs[key]
        _collect_retired_storage()

    @_runtime_locked
    def close(self):
        if self._state == 'closed':
            return
        if self._state != 'open':
            raise RuntimeError(f"Green context pair is {self._state}")
        self._state = 'closing'
        try:
            # Callers retire outer captured graphs first. Primary-role work
            # belongs to the caller's context, which may no longer be current
            # during thread-local teardown.
            if self.primary_ctx_roles:
                current = _ok(cu.cuCtxGetCurrent())
                switch_context = int(current) != int(self.primary_context)
                if switch_context:
                    _ok(cu.cuCtxPushCurrent(self.primary_context))
                try:
                    _ok(cu.cuCtxSynchronize())
                finally:
                    if switch_context:
                        _ok(cu.cuCtxPopCurrent())
            else:
                for green, event in zip(self.green, self.retirement):
                    status = cu.cuGreenCtxRecordEvent(green, event)[0]
                    if status == cu.CUresult.CUDA_SUCCESS:
                        _ok(cu.cuEventSynchronize(event))
                    else:
                        raise RuntimeError(f"cannot retire Green context work: {status}")
            self._clear_graph_cache()
            for graph in self._failed_children:
                graph.close()
            self._failed_children.clear()
            _collect_retired_storage()
            while self._failed_libraries:
                self._failed_libraries[-1].close()
                self._failed_libraries.pop()
            for event in reversed(self.retirement):
                _ok(cu.cuEventDestroy(event))
            for stream in reversed(self.streams):
                _ok(cu.cuStreamDestroy(stream))
            for green in reversed(self.green):
                _ok(cu.cuGreenCtxDestroy(green))
            self.retirement.clear()
            self.streams.clear()
            self.contexts.clear()
            self.green.clear()
        except Exception:
            # A partially retired pair must never be returned from the
            # thread-local cache or used for another launch.
            self._state = 'failed'
            _pin_failed_pair(self)
            raise
        self._state = 'closed'
        _LIVE_PAIRS.pop(id(self), None)


_THREAD_LOCAL = threading.local()
_FAILED_PAIRS: dict[int, GreenContextPair] = {}


def _pin_failed_pair(pair: GreenContextPair) -> None:
    # Failed CUDA retirement has no safe retry contract. Keep live mappings
    # and child resources reachable until process teardown.
    _FAILED_PAIRS[id(pair)] = pair


@_runtime_locked
def _close_runtime():
    global _RUNTIME_STATE
    if _RUNTIME_STATE == 'closed':
        return
    if _RUNTIME_STATE != 'open':
        raise RuntimeError(f"kernel runtime is {_RUNTIME_STATE}")
    _RUNTIME_STATE = 'retiring'
    try:
        for pair in tuple(_LIVE_PAIRS.values()):
            _ok(cu.cuCtxPushCurrent(pair.primary_context))
            try:
                pair.close()
            finally:
                _ok(cu.cuCtxPopCurrent())
        while _LOADED_KERNELS:
            key = next(reversed(_LOADED_KERNELS))
            _LOADED_KERNELS[key].close()
            del _LOADED_KERNELS[key]
    except Exception:
        _RUNTIME_STATE = 'failed'
        raise
    _RUNTIME_STATE = 'closed'


atexit.register(_close_runtime)


class _ThreadContexts:
    def __init__(self):
        self.contexts: dict[tuple[int, int, bool], GreenContextPair] = {}

    def __del__(self):
        # Thread-local destruction runs in the owning thread. Callers must
        # retire any outer captured graphs before this point.
        for pair in self.contexts.values():
            if pair._state == 'open':
                try:
                    pair.close()
                except Exception:
                    _pin_failed_pair(pair)
            elif pair._state == 'failed':
                _pin_failed_pair(pair)


def get_dispatch_sm_config(*, num_ranks: int) -> tuple[int, int]:
    """Resolve the requested and actual dispatch budget without creating a context."""
    requested = _get_requested_dispatch_sms(num_ranks)
    device = _ok(cu.cuDeviceGet(torch.cuda.current_device()))
    _, _, dispatch = _green_resources().get_sm_split(device, requested)
    return requested, dispatch.sm.smCount


def get_context(*, num_ranks: int,
                requested_dispatch_sms: int | None = None) -> GreenContextPair:
    requested_dispatch_sms = _get_requested_dispatch_sms(num_ranks, requested_dispatch_sms)
    current = _ok(cu.cuCtxGetCurrent())
    holder = getattr(_THREAD_LOCAL, 'holder', None)
    if holder is None:
        holder = _THREAD_LOCAL.holder = _ThreadContexts()
    contexts = holder.contexts
    # Calls in different process groups may need different SM budgets even
    # on the same CUDA context. The role mode also changes graph dependencies.
    primary_ctx_roles = _get_env_int('GC_MOE_PRIMARY_CTX', 0) != 0
    key = (int(current), requested_dispatch_sms, primary_ctx_roles)
    if key not in contexts or contexts[key]._state == 'closed':
        contexts[key] = GreenContextPair(
            requested_dispatch_sms=requested_dispatch_sms, num_ranks=num_ranks,
            primary_ctx_roles=primary_ctx_roles)
    elif contexts[key]._state != 'open':
        raise RuntimeError(f"cached Green context pair is {contexts[key]._state}")
    return contexts[key]
