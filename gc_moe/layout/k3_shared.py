"""K3 shared expert本地工作区的容量与字节偏移。"""

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(frozen=True)
class K3SharedLayout:
    """FC1完成标记、BF16 activation和BF16 partial，各区起点按256B对齐。"""

    bf16_bytes: ClassVar[int] = 2
    flag_bytes: ClassVar[int] = 8
    block_n: ClassVar[int] = 128
    block_k: ClassVar[int] = 64
    max_token_tile: ClassVar[int] = 256

    compute_sms: int = 132
    tokens: int = 256
    hidden: int = 7168
    intermediate: int = 3072
    fc1_done: int = field(init=False)
    act: int = field(init=False)
    partial: int = field(init=False)
    nbytes: int = field(init=False)

    def __post_init__(self):
        if (not isinstance(self.compute_sms, int) or self.compute_sms < 32
                or self.compute_sms % 2 or self.hidden != 7168
                or not isinstance(self.tokens, int) or self.tokens < 1
                or not isinstance(self.intermediate, int) or self.intermediate < 64
                or self.intermediate % self.block_k):
            raise ValueError('K3 shared requires an even compute CTA count >=32, positive token capacity, H7168, and I divisible by 64')
        # 设备端ACT/partial使用32位无符号元素索引，最高有效索引须小于2**32。
        if self.tokens * max(self.hidden, self.intermediate) > 2**32:
            raise ValueError('K3 shared ACT/partial capacity must not exceed 2**32 elements')
        offset = 0
        for name, size in (
            ('fc1_done', self.token_tiles * self.fc1_flags_per_tile * self.flag_bytes),
            ('act', self.tokens * self.intermediate * self.bf16_bytes),
            ('partial', self.tokens * self.hidden * self.bf16_bytes),
        ):
            offset = (offset + 255) // 256 * 256
            object.__setattr__(self, name, offset)
            offset += size
        object.__setattr__(self, 'nbytes', (offset + 255) // 256 * 256)

    @property
    def token_tiles(self):
        # n<=256时使用单tile；更长输入按256行分块，最后一块由TMA填零。
        return (self.tokens + self.max_token_tile - 1) // self.max_token_tile

    @property
    def fc1_tiles(self):
        return 2 * self.intermediate // self.block_n

    @property
    def fc1_flags_per_tile(self):
        # 每个M128任务的两CTA各发布一个32通道标记，覆盖该token tile的全部I通道。
        return 2 * self.fc1_tiles

    @property
    def fc2_tiles(self):
        return self.hidden // self.block_n
