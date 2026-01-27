# 不考虑连通保持的覆盖控制场景
import os
import numpy as np
from envs.mpe.multiagent.CoverageWorld import CoverageWorld
from envs.mpe.multiagent.core import Agent, Landmark
from envs.mpe.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self, num_agents=4, num_pois=20, r_cover=0.25, r_comm=0.5, comm_r_scale=0.9, comm_force_scale=0.5, use_comm_reward=True):
        # 【新增use_comm_reward】是否启用连通性奖励（baseline / multi-objective 开关）

        # agents的数量, 起飞位置, poi的数量和起飞位置
        self.num_agents = num_agents
        self.num_pois = num_pois
        self.pos_pois = np.load(
            os.path.join(os.path.dirname(__file__), "pos_pois.npy")
        )[0:num_pois, :]
        # self.pos_pois = np.random.uniform(-1, 1, (num_pois, 2))

        # =========================
        # 覆盖与通信参数
        # =========================
        self.r_cover = r_cover
        self.r_comm = r_comm

        self.use_comm_reward = use_comm_reward # 【新增】多目标奖励开关（是否考虑连通性）
        self.r_cover_rate = 5.0  # 【新增】覆盖率奖励权重

        # =========================
        # 物理与能量参数（原有）
        # =========================
        self.size = 0.02
        self.m_energy = 5.0

        # =========================
        # 奖励权重（原有）
        # =========================
        self.rew_cover = 75.0 # 单个 PoI 完成奖励
        self.rew_done = 1500.0 # 全部 PoI 完成奖励
        # 未连通惩罚
        self.rew_unconnect = -0.0

        self.rew_out = -100

        # =========================
        # 连通保持相关参数（原有）
        # =========================
        self.comm_r_scale = comm_r_scale  # r_comm * comm_force_scale = 计算聚合力时的通信半径
        self.comm_force_scale = comm_force_scale  # 连通保持聚合力的倍数

    def make_world(self):
        world = CoverageWorld()
        # world.bb = 1.2
        # world.boundary = [np.array([world.bb, 0]), np.array([-world.bb, 0]),
        #                   np.array([0, world.bb]), np.array([0, -world.bb])]

        world.collaborative = True
        num_agents = 4
        num_landmarks = 20

        world.agents = [Agent() for _ in range(num_agents)]  # 代表UAV, size为覆盖面积
        world.landmarks = [Landmark() for _ in range(num_landmarks)]

        for i, agent in enumerate(world.agents):
            agent.name = "agent_%d" % i
            agent.collide = False
            agent.silent = True
            agent.size = self.size
            agent.r_cover = self.r_cover
            agent.r_comm = self.r_comm
            agent.max_speed = 0.5
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "poi_%d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = self.size
            landmark.m_energy = self.m_energy

        self.reset_world(world)
        return world

    def reset_world(self, world):
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.05, 0.15, 0.05])
            agent.cover_color = np.array([0.05, 0.25, 0.05])
            agent.comm_color = np.array([0.05, 0.35, 0.05])
            agent.state.p_pos = np.zeros(world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            # landmark.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            landmark.state.p_pos = self.pos_pois[i, :]
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.energy = 0.0
            landmark.done, landmark.just = False, False

    def reward(self, agent, world):
        """
        多目标奖励：
            1) 覆盖距离奖励
            2) 覆盖完成度奖励
            3) 全局完成度奖励
            4) 边界惩罚
            5) 连通性奖励（新增）
        """
        rew = 0.0
        # =========================
        # 1. 覆盖距离奖励（原有）
        # =========================
        for poi in world.landmarks:
            if not poi.done:
                dists = [
                    np.linalg.norm(ag.state.p_pos - poi.state.p_pos)
                    for ag in world.agents
                ]
                rew -= min(dists)
            elif poi.just:
                rew += self.rew_cover
                poi.just = False
        # =========================
        # 2. 全部覆盖完成奖励（原有）
        # =========================
        if all([poi.done for poi in world.landmarks]):
            rew += self.rew_done
        # =========================
        # 3. 出界惩罚（原有）
        # =========================
        for ag in world.agents:
            abs_pos = np.abs(ag.state.p_pos)
            rew += np.sum(abs_pos[abs_pos > 1] - 1) * self.rew_out
            if (abs_pos > 1.5).any():
                rew += self.rew_out
        # =========================
        # 4. 连通性奖励（软约束）
        # =========================
        if self.use_comm_reward:
            adj = world.adj_mat
            n = adj.shape[0]
            connect_ratio = np.sum(adj) / (n * (n - 1) + 1e-6)
            rew += self.r_comm * connect_ratio
        # =========================
        # 5. 覆盖率奖励（协同指标）
        # =========================
        rew += self.r_cover_rate * world.coverage_rate

        return rew

    def observation(self, agent, world):
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        pos_pois = []
        for poi in world.landmarks:
            pos_pois.append(poi.state.p_pos - agent.state.p_pos)
            pos_pois.append([poi.energy, poi.m_energy, poi.done])
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + pos_pois)

    def done(self, agent, world):
        for ag in world.agents:
            abs_pos = np.abs(ag.state.p_pos)
            if (abs_pos > 1.5).any():
                return True
        return all([poi.done for poi in world.landmarks])














