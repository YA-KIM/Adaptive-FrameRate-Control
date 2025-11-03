# input vector가 list인지 numpy인지에 따라 코드 수정이 약간 필요함함
import numpy as np

class Moment():
    def __init__(self, vector, obj_id):
        # 항상 (1, 9) 크기 유지
        vector = np.array(vector).reshape(1, -1)
        if vector.shape[1] == 8:
            vector = np.hstack((vector, np.zeros((1, 1))))  # 각속도(ang_vel) 추가
        self.current_vector = vector
        self.obj_id = obj_id
        self.previous_vector = None


    def copy(self):
        copyone = Moment(self.current_vector.copy(),self.obj_id)
        copyone.previous_vector = self.previous_vector
        return copyone

    def update(self, New_Vector):
        self.previous_vector=self.current_vector
        self.current_vector=New_Vector if isinstance(New_Vector, np.ndarray) else np.array(New_Vector).reshape(1, -1)
        
    def __repr__(self):
        """객체 정보를 보기 쉽게 출력"""
        return f"Moment(obj_id={self.obj_id}, current_vector={self.current_vector}, previous_vector={self.previous_vector})"

class History_Supervisor():
    def __init__(self, History_Length,  max_inactive_frames=10):
        self.His_Len = History_Length
        self.max_inactive_frames = max_inactive_frames
        self.histories = {} #ID별 history를 모아둘 Dictionary onj_id:History 형태로 존재
        self.last_updated = {}

    def _initailize_history(self, obj_id):
        #새로운 ID 일 시에, current vector가 0인 moment들로 초기화화
        zero_vector = np.zeros((1, 8))
        self.histories[obj_id] = [Moment(zero_vector, obj_id) for _ in range(self.His_Len)]
        self.last_updated[obj_id] = 0
    
    def update(self, obj_id, vector, current_frame):
        if vector is None:
            vector = np.zeros((1, 9), dtype=np.float32) 

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector).reshape(1,-1)

        if obj_id not in self.histories: # obj_id Key가 없으면 해당 키에 initail history할당당
            self._initailize_history(obj_id)
        last_moment = self.histories[obj_id][-1].copy() #obj_id에 해당하는 객체의 마지막 moment
        last_moment.update(vector)

        # 가장 오래된 데이터 제거하고 새 데이터 추가
        self.histories[obj_id].pop(0)
        self.histories[obj_id].append(last_moment)

        # 최근 업데이트 프레임 기록
        self.last_updated[obj_id] = current_frame

        # 오래된 객체 삭제 (자동으로 수행)
        self.prune_old_entries(current_frame)

    def get_state_history(self, obj_id):
        history = self.histories.get(obj_id, [])
        # 아마 list[numpy,numpy,,,]일거라..
        # vector 형태로 history를 반환환
        if not history:  # 🔹 빈 리스트일 경우
            return np.zeros((1, 9), dtype=np.float32) 
        
        return np.vstack([moment.current_vector for moment in self.histories.get(obj_id, [])])

    def get_moment_history(self,obj_id):
        # moment 형태로 history를 반환환
        # index 9번이 가장 최신. CNN에 들어가기 위해선 변환해줘야함함
        return self.histories.get(obj_id, [])

    def prune_old_entries(self, current_frame):
        """최근 `max_inactive_frames` 동안 업데이트되지 않은 객체 삭제"""
        inactive_ids = [obj_id for obj_id, last_frame in self.last_updated.items()
                        if current_frame - last_frame > self.max_inactive_frames]

        for obj_id in inactive_ids:
            del self.histories[obj_id]
            del self.last_updated[obj_id]
    
    def clear(self):
        """전체 히스토리 초기화"""
        self.histories.clear()
        self.last_updated.clear()

    def __repr__(self):
        if not self.histories:
            return "Initialized!"
        return "\n".join([f"ID {obj_id}: {history}" for obj_id, history in self.histories.items()])
    
    def __getitem__(self, obj_id):
        return self.histories.get(obj_id, [])
    
        