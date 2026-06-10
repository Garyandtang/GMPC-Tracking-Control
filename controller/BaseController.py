from abc import ABC, abstractmethod

class BaseController(ABC):
    def __init__(self, ref_traj_config=None):
        self.solve_time = 0.0
        if ref_traj_config:
            self.set_ref_traj(ref_traj_config)
        
    def set_ref_traj(self, traj_config):
        traj_generator = TrajGenerator(traj_config)
        self.ref_state, self.ref_control, self.dt = traj_generator.get_traj()
        self.nTraj = self.ref_state.shape[1]
    
    @abstractmethod
    def solve(self, state, t):
        pass
    
    def get_solve_time(self):
        return self.solve_time
    
    def set_control_bound(self, v_min=-4, v_max=4, w_min=-4, w_max=4):
        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max
    
    def saturate_control(self, vel_cmd):
        return np.clip(vel_cmd, [self.v_min, self.w_min], [self.v_max, self.w_max])
