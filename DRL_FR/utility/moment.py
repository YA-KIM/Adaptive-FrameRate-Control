from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Deque, List, Optional
import numpy as np

VEC_DIM = 9  # [cx, cy, h, w, vx, vy, ax, ay, ang_vel]

def _as_row_vec9(x) -> np.ndarray:
    """
    임의 입력(list/np.ndarray/shape (9,) or (1,9) or (1,8))을
    (1, 9) float32로 강제 변환. (1,8)은 마지막 ang_vel=0 추가.
    """
    arr = np.asarray(x, dtype=np.float32).reshape(1, -1)
    if arr.shape[1] == 8:
        arr = np.hstack([arr, np.zeros((1, 1), dtype=np.float32)])
    if arr.shape != (1, VEC_DIM):
        raise ValueError(f"vector must be shape (1,{VEC_DIM}) or (1,8); got {arr.shape}")
    return arr

@dataclass
class Moment:
    """단일 시점 상태 스냅샷 (current/previous 모두 (1,9) float32)."""
    obj_id: int | float
    current_vector: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((1, VEC_DIM), np.float32))
    previous_vector: Optional[np.ndarray] = field(repr=False, default=None)

    def __init__(self, vector, obj_id):
        self.obj_id = obj_id
        self.current_vector = _as_row_vec9(vector)
        self.previous_vector = None

    def copy(self) -> "Moment":
        m = Moment(self.current_vector.copy(), self.obj_id)
        m.previous_vector = None if self.previous_vector is None else self.previous_vector.copy()
        return m

    def update(self, new_vector) -> None:
        """previous <- current, current <- new"""
        self.previous_vector = self.current_vector
        self.current_vector = _as_row_vec9(new_vector)

    def __repr__(self) -> str:
        return (f"Moment(obj_id={self.obj_id}, "
                f"current_vector=shape{self.current_vector.shape}, "
                f"previous_vector={'None' if self.previous_vector is None else 'set'})")

class History_Supervisor:
    """
    객체별 최근 히스토리를 고정 길이로 보관.
    - histories[obj_id] : deque[Moment] (maxlen=His_Len)
    - last_updated[obj_id] : 최근 프레임 index
    """
    def __init__(self, History_Length: int, max_inactive_frames: int = 10):
        if History_Length <= 0:
            raise ValueError("history_length must be positive")
        self.His_Len = History_Length
        self.max_inactive_frames = max_inactive_frames
        self.histories: Dict[int | float, Deque[Moment]] = {}
        self.last_updated: Dict[int | float, int] = {}

    def _initialize_history(self, obj_id):
        """새 obj_id 히스토리를 zero-vector로 채워 초기화."""
        zero = np.zeros((1, VEC_DIM), dtype=np.float32)
        dq: Deque[Moment] = deque(maxlen=self.His_Len)
        for _ in range(self.His_Len):
            dq.append(Moment(zero, obj_id))
        self.histories[obj_id] = dq
        self.last_updated[obj_id] = -10**9  # 아주 과거로 설정

    def update(self, obj_id, vector, current_frame: int) -> None:
        """
        obj_id의 최근 히스토리를 갱신.
        vector가 None이면 zero-vector로 대체.
        """
        if vector is None:
            vector = np.zeros((1, VEC_DIM), np.float32)

        vec = _as_row_vec9(vector)

        if obj_id not in self.histories:
            self._initialize_history(obj_id)

        # 마지막 상태 복사 후 갱신하여 push
        last_moment = self.histories[obj_id][-1].copy()
        last_moment.update(vec)
        self.histories[obj_id].append(last_moment)

        self.last_updated[obj_id] = int(current_frame)
        self.prune_old_entries(current_frame)

    def get_state_history(self, obj_id) -> np.ndarray:
        """
        학습/정책 네트워크 입력용 상태 배열 반환.
        항상 shape (His_Len, 9), float32 보장.
        """
        if obj_id not in self.histories:
            # 없는 경우도 일관된 shape 반환
            return np.zeros((self.His_Len, VEC_DIM), dtype=np.float32)
        dq = self.histories[obj_id]
        arr = np.vstack([m.current_vector for m in dq])           # (His_Len, 9) *이미 (1,9)씩*
        return arr.astype(np.float32, copy=False)

    def get_moment_history(self, obj_id) -> List[Moment]:
        """Moment 리스트(오래됨→최신) 반환. 없으면 빈 리스트."""
        if obj_id not in self.histories:
            return []
        return list(self.histories[obj_id])

    def prune_old_entries(self, current_frame: int) -> None:
        """최근 max_inactive_frames 동안 업데이트 없는 obj를 제거."""
        cutoff = int(current_frame) - self.max_inactive_frames
        stale = [oid for oid, t in self.last_updated.items() if t < cutoff]
        for oid in stale:
            self.histories.pop(oid, None)
            self.last_updated.pop(oid, None)

    def clear(self) -> None:
        """전체 히스토리 초기화."""
        self.histories.clear()
        self.last_updated.clear()

    def __repr__(self) -> str:
        if not self.histories:
            return "HistorySupervisor(empty)"
        lines = []
        for oid, dq in self.histories.items():
            lines.append(f"ID {oid}: len={len(dq)}, last_frame={self.last_updated.get(oid,'?')}")
        return "\n".join(lines)

    def __getitem__(self, obj_id) -> List[Moment]:
        return self.get_moment_history(obj_id)