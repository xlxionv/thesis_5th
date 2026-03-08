import numpy as np

import gym
from gym import spaces

from onpolicy.envs.mpe.multi_discrete import MultiDiscrete


class BoschEnv(object):
    """
    Parallel-line production + maintenance environment for multi-agent RL.

    Agents:
        agent 0 : lot sizing & line allocation agent (global)
        agents 1-6 : machine agents, one per line
    """

    def __init__(self, args):
        # Core configuration (with sensible defaults if attributes are missing)
        self.num_lines = getattr(args, "num_lines", 6)
        self.num_products = getattr(args, "num_products", 3)
        # Number of decision periods (planning horizon)
        num_periods = getattr(args, "num_periods", None)
        if num_periods is None:
            num_periods = getattr(args, "episode_length", 24)
        self.num_periods = int(num_periods)

        # Maximum number of machine actions (micro-steps) per period
        self.max_actions_per_period = int(
            getattr(args, "max_actions_per_period", 8)
        )

        # Capacity and demand parameters
        self.capacity_per_line = float(getattr(args, "capacity_per_line", 100.0))
        self.max_lot_size = int(getattr(args, "max_lot_size", 10))

        # Cost parameters
        self.holding_cost = float(getattr(args, "holding_cost", 1.0))
        self.backlog_cost = float(getattr(args, "backlog_cost", 10.0))
        self.production_cost = float(getattr(args, "production_cost", 1.0))
        self.setup_cost = float(getattr(args, "setup_cost", 2.0))
        self.pm_cost = float(getattr(args, "pm_cost", 20.0))
        self.cm_cost = float(getattr(args, "cm_cost", 40.0))
        self.alpha_cost_weight = float(getattr(args, "alpha_cost_weight", 0.1))

        # Simple exponential hazard-rate style degradation
        self.hazard_rate = float(getattr(args, "hazard_rate", 1e-3))

        # Time-related parameters (capacity is in "hours" or generic time units)
        self.pm_time = float(getattr(args, "pm_time", 0.0))
        self.cm_time = float(getattr(args, "cm_time", 0.0))

        # Per-product processing times and mean demand (can be scalars or comma-separated lists)
        self.processing_time = self._get_array_arg(
            args, "processing_time", self.num_products, default=1.0
        )
        self.mean_demand = self._get_array_arg(
            args, "mean_demand", self.num_products, default=10.0
        )

        # Sequence-dependent setup cost/time matrices and heterogeneous production costs
        base_setup_cost = float(getattr(args, "setup_cost", 2.0))
        base_setup_time = float(getattr(args, "setup_time", 0.0))
        self.setup_cost_matrix = self._get_matrix_arg(
            args,
            "setup_cost_matrix",
            (self.num_products, self.num_products),
            default=base_setup_cost,
        )
        self.setup_time_matrix = self._get_matrix_arg(
            args,
            "setup_time_matrix",
            (self.num_products, self.num_products),
            default=base_setup_time,
        )

        base_prod_cost = float(getattr(args, "production_cost", 1.0))
        self.production_cost_matrix = self._get_matrix_arg(
            args,
            "production_cost_matrix",
            (self.num_lines, self.num_products),
            default=base_prod_cost,
        )

        # Product-line eligibility restrictions (1 if line can produce product, else 0)
        self.line_eligibility = self._get_matrix_arg(
            args,
            "eligibility_matrix",
            (self.num_lines, self.num_products),
            default=1.0,
        )
        self.line_eligibility = (self.line_eligibility > 0.5).astype(np.float32)

        # Time / episode tracking (micro-steps within periods)
        self.current_step = 0
        self.period_index = 0
        self.step_in_period = 0

        # Agents
        # agent 0 : lot sizing & allocation
        # agents 1..num_lines : machine agents
        self.num_agents = 1 + self.num_lines

        # Observation layout (same length for all agents)
        # [inventory (P),
        #  backlog (P),
        #  remaining_demand (P),
        #  remaining_periods (1),
        #  line_availability (L),
        #  line_setup (L one-hot over products, flattened),
        #  ages (L),
        #  local_line_id_one_hot (L),
        #  padding ...]
        self.obs_dim = (
            3 * self.num_products
            + 1
            + self.num_lines
            + self.num_lines * self.num_products
            + self.num_lines
            + self.num_lines
        )

        high = np.full(self.obs_dim, np.inf, dtype=np.float32)
        low = -high
        self.observation_space = [
            spaces.Box(low=low, high=high, dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        # Centralized observation = concatenation of all agent observations
        share_high = np.full(self.obs_dim * self.num_agents, np.inf, dtype=np.float32)
        share_low = -share_high
        self.share_observation_space = [
            spaces.Box(low=share_low, high=share_high, dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        # Action spaces
        # Agent 0: MultiDiscrete over lot sizes per (line, product)
        # Each dim k in [0, max_lot_size]
        lot_dims = [[0, self.max_lot_size]] * (self.num_lines * self.num_products)
        agent0_act = MultiDiscrete(lot_dims)

        # Machine agents: Discrete
        #   0..num_products-1 => process product
        #   num_products      => perform PM
        #   num_products + 1  => end shift
        machine_act = spaces.Discrete(self.num_products + 2)

        self.action_space = [agent0_act] + [machine_act for _ in range(self.num_lines)]

        # Internal state
        self.rng = np.random.RandomState(getattr(args, "seed", 1))

        self._build_demand_profile()
        self._reset_state()

    def _get_array_arg(self, args, name, length, default):
        """
        Helper to read scalar or comma-separated list arguments into
        a 1D float array of shape [length].
        """
        raw = getattr(args, name, None)
        if raw is None:
            return np.ones(length, dtype=np.float32) * float(default)

        if isinstance(raw, (list, np.ndarray)):
            arr = np.asarray(raw, dtype=np.float32)
        else:
            parts = str(raw).split(",")
            arr = np.array([float(p) for p in parts], dtype=np.float32)

        if arr.size == 1:
            arr = np.repeat(arr, length)

        if arr.size != length:
            raise ValueError(
                f"Argument {name} expects length {length}, got {arr.size}."
            )
        return arr

    def _get_matrix_arg(self, args, name, shape, default):
        """
        Helper to read scalar or comma-separated list arguments into
        a 2D float array with given shape.
        """
        raw = getattr(args, name, None)
        rows, cols = shape
        if raw is None:
            return np.ones((rows, cols), dtype=np.float32) * float(default)

        if isinstance(raw, (list, np.ndarray)):
            arr = np.asarray(raw, dtype=np.float32)
        else:
            parts = str(raw).split(",")
            arr = np.array([float(p) for p in parts], dtype=np.float32)

        if arr.size == 1:
            arr = np.repeat(arr, rows * cols)

        if arr.size != rows * cols:
            raise ValueError(
                f"Argument {name} expects {rows * cols} values, got {arr.size}."
            )
        return arr.reshape(rows, cols)

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------
    def seed(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)

    def reset(self):
        self._reset_state()
        self._start_new_period()
        return self._build_observations()

    def step(self, actions_env):
        """
        One environment step corresponds to a single micro-step within
        a period. Each period consists of:
            - Manager phase (step_in_period == 0): Agent 0 allocates lots.
            - Worker phases (subsequent micro-steps): machine agents act
              repeatedly until capacity or queue is exhausted.

        :param actions_env: list of per-agent one-hot / multi-one-hot actions.
        """
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        done = False

        manager_phase = self.step_in_period == 0

        if manager_phase:
            self._manager_step(actions_env)
        else:
            self._machines_step(actions_env)

        # Advance micro-step counters
        self.step_in_period += 1
        self.current_step += 1

        # Decide whether to end this period
        end_of_period = (
            self.step_in_period >= self.max_actions_per_period
            or self._period_effectively_over()
        )

        if end_of_period:
            day_rewards = self._end_period()
            rewards += day_rewards
            self.period_index += 1
            self.step_in_period = 0

            if self.period_index < self.num_periods:
                self._start_new_period()
            else:
                done = True

        obs = self._build_observations()
        dones = [done for _ in range(self.num_agents)]
        infos = [{} for _ in range(self.num_agents)]

        return obs, rewards, dones, infos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_demand_profile(self):
        """
        Simple deterministic demand: constant per product per period.
        Can be replaced with real Bosch parameters or samples.
        """
        base_demand = self.mean_demand.astype(np.float32)
        self.demand = np.tile(base_demand[None, :], (self.num_periods, 1))

    def _reset_state(self):
        # Episode-level counters
        self.current_step = 0
        self.period_index = 0
        self.step_in_period = 0

        # Inventory & backlog per product
        self.inventory = np.zeros(self.num_products, dtype=np.float32)
        self.backlog = np.zeros(self.num_products, dtype=np.float32)

        # Machine line states
        self.ages = np.zeros(self.num_lines, dtype=np.float32)
        # -1 means "no previous product"
        self.line_setup = np.full(self.num_lines, -1, dtype=np.int32)

        # Queues per line and product (assigned by manager, consumed by machines)
        self.queue = np.zeros(
            (self.num_lines, self.num_products), dtype=np.float32
        )

        # Per-period aggregates
        self.remaining_capacity = np.zeros(self.num_lines, dtype=np.float32)
        self.line_done = np.zeros(self.num_lines, dtype=bool)
        self.period_produced_per_line = np.zeros(
            (self.num_lines, self.num_products), dtype=np.float32
        )
        self.period_produced_per_product = np.zeros(
            self.num_products, dtype=np.float32
        )
        self.period_setup_costs = np.zeros(self.num_lines, dtype=np.float32)
        self.period_pm_costs = np.zeros(self.num_lines, dtype=np.float32)
        self.period_cm_costs = np.zeros(self.num_lines, dtype=np.float32)

    def _start_new_period(self):
        # Reset per-period capacity, flags, and aggregates, but keep
        # inventory, backlog, ages and queue.
        self.remaining_capacity[:] = self.capacity_per_line
        self.line_done[:] = False
        self.period_produced_per_line.fill(0.0)
        self.period_produced_per_product.fill(0.0)
        self.period_setup_costs.fill(0.0)
        self.period_pm_costs.fill(0.0)
        self.period_cm_costs.fill(0.0)

    def _manager_step(self, actions_env):
        """
        Agent 0 chooses lot allocations once per period.
        """
        # Decode agent 0 action (lot sizes MultiDiscrete as concatenated one-hot)
        lot_matrix = self._decode_agent0_action(actions_env[0])

        # Enforce product-line eligibility: ineligible lots are discarded.
        lot_matrix = lot_matrix * self.line_eligibility

        # Add allocated lots into each line's queue.
        self.queue += lot_matrix

    def _machines_step(self, actions_env):
        """
        Machine agents act given current queues and remaining capacity.
        """
        pm_index = self.num_products
        end_index = self.num_products + 1

        for line_idx in range(self.num_lines):
            if self.line_done[line_idx]:
                continue

            agent_id = 1 + line_idx
            a_vec = np.asarray(actions_env[agent_id], dtype=np.float32)
            act_idx = int(np.argmax(a_vec))

            # Explicit "End Shift"
            if act_idx == end_index:
                self.line_done[line_idx] = True
                continue

            # Preventive maintenance
            if act_idx == pm_index:
                if self.pm_time > 0.0 and self.remaining_capacity[line_idx] > 0.0:
                    used = min(self.pm_time, self.remaining_capacity[line_idx])
                    self.remaining_capacity[line_idx] -= used
                self.period_pm_costs[line_idx] += self.pm_cost
                self.ages[line_idx] = 0.0
                # PM does not automatically end the shift; further actions may follow.
                continue

            # Produce a product
            if act_idx < 0 or act_idx >= self.num_products:
                # Invalid index; treat as end of shift.
                self.line_done[line_idx] = True
                continue

            # Check eligibility
            if self.line_eligibility[line_idx, act_idx] < 0.5:
                # Ineligible product on this line; end shift to discourage this choice.
                self.line_done[line_idx] = True
                continue

            # How much is waiting in this line's queue for this product?
            requested_qty = float(self.queue[line_idx, act_idx])
            if requested_qty <= 0.0:
                # Nothing left to process; end shift for this line.
                self.line_done[line_idx] = True
                continue

            last_prod = int(self.line_setup[line_idx])
            setup_time = 0.0
            setup_cost = 0.0
            if last_prod >= 0 and last_prod != act_idx:
                setup_time = float(self.setup_time_matrix[last_prod, act_idx])
                setup_cost = float(self.setup_cost_matrix[last_prod, act_idx])

            proc_time_per_unit = float(self.processing_time[act_idx])
            if proc_time_per_unit <= 0.0:
                # Cannot process this product.
                self.line_done[line_idx] = True
                continue

            available_for_proc = self.remaining_capacity[line_idx] - setup_time
            if available_for_proc <= 0.0:
                # Not enough time even to pay setup; end shift.
                self.line_done[line_idx] = True
                continue

            max_qty_cap = int(available_for_proc // proc_time_per_unit)
            if max_qty_cap <= 0:
                # Not enough capacity to process a single unit.
                self.line_done[line_idx] = True
                continue

            qty = min(int(requested_qty), max_qty_cap)
            if qty <= 0:
                self.line_done[line_idx] = True
                continue

            # Time consumed this micro-step on this line
            time_used = qty * proc_time_per_unit + setup_time
            self.remaining_capacity[line_idx] = max(
                0.0, self.remaining_capacity[line_idx] - time_used
            )

            # Update queue and production aggregates
            self.queue[line_idx, act_idx] = max(
                0.0, self.queue[line_idx, act_idx] - qty
            )
            self.period_produced_per_line[line_idx, act_idx] += qty
            self.period_produced_per_product[act_idx] += qty

            # Sequence-dependent setup cost if there was a switch
            if setup_time > 0.0 or setup_cost > 0.0:
                self.period_setup_costs[line_idx] += setup_cost
                self.line_setup[line_idx] = act_idx

            # Age increases with actual runtime (excluding setup)
            self.ages[line_idx] += qty * proc_time_per_unit

            # If no capacity remains, end shift for this line.
            if self.remaining_capacity[line_idx] <= 0.0:
                self.line_done[line_idx] = True

    def _period_effectively_over(self):
        """
        Simple heuristic to end a period early:
        - All lines have ended their shift, OR
        - There is no remaining capacity on any line, OR
        - There is no work left in any queue.
        """
        if np.all(self.line_done):
            return True
        if np.all(self.remaining_capacity <= 0.0):
            return True
        if np.all(self.queue <= 0.0):
            return True
        return False

    def _end_period(self):
        """
        Compute per-period costs and rewards, update inventory/backlog,
        and return a reward vector of length num_agents for this day.
        """
        # Inventory / backlog cost update based on total production this period.
        inv_cost, backlog_cost = self._update_inventory_and_backlog(
            self.period_produced_per_product
        )

        # Setup and PM costs are already accumulated per line.
        setup_cost_total = float(np.sum(self.period_setup_costs))
        pm_cost_total = float(np.sum(self.period_pm_costs))

        # Expected CM cost from current ages
        expected_failures = self.hazard_rate * self.ages
        self.period_cm_costs = expected_failures * self.cm_cost
        expected_cm_cost_total = float(np.sum(self.period_cm_costs))

        # Heterogeneous production cost per line and product
        prod_cost_total = float(
            np.sum(self.period_produced_per_line * self.production_cost_matrix)
        )

        total_period_cost = (
            inv_cost
            + backlog_cost
            + setup_cost_total
            + pm_cost_total
            + expected_cm_cost_total
            + prod_cost_total
        )

        rewards = np.zeros(self.num_agents, dtype=np.float32)

        # Agent 0: lot-sizing & allocation
        rewards[0] = -float(inv_cost + backlog_cost) + self.alpha_cost_weight * (
            -float(total_period_cost)
        )

        # Machine agents: per line local costs + shared global term
        for line_idx in range(self.num_lines):
            local_cost = (
                self.period_setup_costs[line_idx]
                + self.period_pm_costs[line_idx]
                + self.period_cm_costs[line_idx]
            )
            rewards[1 + line_idx] = -float(local_cost) + self.alpha_cost_weight * (
                -float(total_period_cost)
            )

        return rewards

    def _update_inventory_and_backlog(self, produced):
        # produced is total quantity per product in this period
        t = min(self.period_index, self.num_periods - 1)
        period_demand = self.demand[t]

        inv_cost = 0.0
        backlog_cost = 0.0

        for p in range(self.num_products):
            total_demand = period_demand[p] + self.backlog[p]
            total_supply = self.inventory[p] + produced[p]

            if total_supply >= total_demand:
                self.inventory[p] = total_supply - total_demand
                self.backlog[p] = 0.0
            else:
                self.inventory[p] = 0.0
                self.backlog[p] = total_demand - total_supply

            inv_cost += self.holding_cost * self.inventory[p]
            backlog_cost += self.backlog_cost * self.backlog[p]

        return inv_cost, backlog_cost

    def _decode_agent0_action(self, action_vec):
        """
        Convert concatenated one-hot representation back into a
        [num_lines, num_products] lot-size matrix.
        """
        one_hot = np.asarray(action_vec, dtype=np.float32).ravel()

        num_dims = self.num_lines * self.num_products
        block_size = self.max_lot_size + 1  # values 0..max_lot_size

        expected_len = num_dims * block_size
        if one_hot.size != expected_len:
            raise ValueError(
                f"Agent 0 action length {one_hot.size} does not match "
                f"expected {expected_len} for "
                f"{num_dims} dims and block_size {block_size}."
            )

        lots = np.zeros(num_dims, dtype=np.float32)
        offset = 0
        for d in range(num_dims):
            segment = one_hot[offset : offset + block_size]
            idx = int(np.argmax(segment))
            lots[d] = float(idx)
            offset += block_size

        lot_matrix = lots.reshape(self.num_lines, self.num_products)
        return lot_matrix

    def _build_observations(self):
        """
        Build per-agent observations with a fixed-length layout.
        """
        remaining_periods = self.num_periods - self.period_index

        # Shared pieces
        inv = self.inventory.astype(np.float32)
        back = self.backlog.astype(np.float32)
        remaining_demand = (
            np.sum(self.demand[self.period_index :], axis=0)
            if self.period_index < self.num_periods
            else np.zeros_like(inv)
        )

        # Line availability = always 1 in this simplified model
        line_availability = np.ones(self.num_lines, dtype=np.float32)

        # Line setup one-hot per line over products
        line_setup_oh = np.zeros(
            (self.num_lines, self.num_products), dtype=np.float32
        )
        for l in range(self.num_lines):
            idx = int(self.line_setup[l])
            if 0 <= idx < self.num_products:
                line_setup_oh[l, idx] = 1.0
        line_setup_flat = line_setup_oh.reshape(-1)

        ages = self.ages.astype(np.float32)

        obs_all = []
        for agent_id in range(self.num_agents):
            vec = np.zeros(self.obs_dim, dtype=np.float32)
            pos = 0

            # Inventory
            vec[pos : pos + self.num_products] = inv
            pos += self.num_products

            # Backlog
            vec[pos : pos + self.num_products] = back
            pos += self.num_products

            # Remaining demand
            vec[pos : pos + self.num_products] = remaining_demand
            pos += self.num_products

            # Remaining periods
            vec[pos] = float(remaining_periods)
            pos += 1

            # Line availability
            vec[pos : pos + self.num_lines] = line_availability
            pos += self.num_lines

            # Line setup (flattened one-hot)
            vec[pos : pos + self.num_lines * self.num_products] = line_setup_flat
            pos += self.num_lines * self.num_products

            # Ages
            vec[pos : pos + self.num_lines] = ages
            pos += self.num_lines

            # Local line id one-hot (for machine agents only)
            line_id_oh = np.zeros(self.num_lines, dtype=np.float32)
            if agent_id > 0:
                line_idx = agent_id - 1
                if 0 <= line_idx < self.num_lines:
                    line_id_oh[line_idx] = 1.0
            vec[pos : pos + self.num_lines] = line_id_oh
            pos += self.num_lines

            obs_all.append(vec)

        return np.asarray(obs_all, dtype=np.float32)

    # Optional compatibility with env_wrappers rendering
    def render(self, mode="human"):
        return None

