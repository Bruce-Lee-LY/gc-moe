"""SM90 role geometry and shared-memory budgets for resident compute CTAs."""
from dataclasses import dataclass, replace
from functools import lru_cache


@dataclass(frozen=True)
class SM90RoleConfig:
    grids: tuple[int, int, int]
    tile_m: int = 32
    threads: tuple[int, int, int] = (256, 256, 256)
    # Dispatch allocates a dynamic shared uint32 counter; the graph must reserve it.
    smem: tuple[int, int, int] = (128, 65792, 0)


@lru_cache(maxsize=256)
def _get_sm90_role_config(config, dispatch_sms: int, compute_sms: int):
    small_pipelined = 0 < config.bound <= 2
    pipelined = small_pipelined or config.bound == 0 or config.bound >= 32
    # Common public bound controls geometry only. Every received row is still
    # processed by the kernel's complete M-chunk loop, including skewed loads.
    expected_rows = (config.bound or config.capacity) * config.ranks * config.topk / config.experts
    if pipelined:
        tile_m = 8 if small_pipelined else (16 if expected_rows <= 12 else (32 if expected_rows <= 24 else 64))
    else:
        tile_m = 8 if config.bound <= 16 else 16
    workers = (1 if tile_m == 64 else 2) if pipelined else (3 if tile_m == 8 else 2)
    if pipelined:
        stage_bytes = (17408 + tile_m * 132 + 1023) // 1024 * 1024
        stages = 5 if tile_m == 64 else (3 if tile_m == 32 else 4)
        compute_smem = stages * stage_bytes + 1024 + tile_m * 1024 + (0 if tile_m == 64 else 256)
    else:
        compute_smem = 53376
    direct_scan = 0 < config.bound <= 2
    combine_bound = config.bound or config.capacity
    combine_grid = min(compute_sms * 4, max(1, (combine_bound * config.hidden + 2047) // 2048))
    return SM90RoleConfig((config.experts // config.ranks if direct_scan else dispatch_sms * 4,
                           compute_sms * workers, combine_grid),
                          tile_m=tile_m,
                          threads=(128 if direct_scan else 256, 384 if pipelined else 256, 384),
                          smem=(max(128, ((2 * config.ranks + 2) * 4 + 127) // 128 * 128), compute_smem, 0))


def get_sm90_role_config(config, context):
    return _get_sm90_role_config(config, context.dispatch_sms, context.compute_sms)


@dataclass(frozen=True)
class RoutedRoleConfig:
    schedule: SM90RoleConfig
    n_factor: int


@lru_cache(maxsize=256)
def get_routed_role_config(config, schedule):
    if (config.shared_experts > 0 and schedule.threads[1] == 256
            and schedule.tile_m == 8 and config.hidden % 512 == 0
            and config.intermediate % 256 == 0):
        return RoutedRoleConfig(
            replace(schedule, smem=(schedule.smem[0], 73856, schedule.smem[2])), 2)
    return RoutedRoleConfig(schedule, 1)


@lru_cache(maxsize=256)
def get_shared_role_config(schedule):
    smem = schedule.smem
    if schedule.threads[1] != 384:
        stage_stride = (36864 + schedule.tile_m * 144 + 1023) // 1024 * 1024
        shared_smem = (2 * stage_stride + 128 + 1023) // 1024 * 1024
        smem = (smem[0], shared_smem, smem[2])
    return replace(schedule, threads=(schedule.threads[0], 256, schedule.threads[2]),
                   smem=smem)
