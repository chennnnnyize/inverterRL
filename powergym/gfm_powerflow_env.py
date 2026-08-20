"""Standalone AC power flow and grid-forming inverter Gym environment.

This file intentionally has no dependency on OpenDSS or the rest of PowerGym.
It contains:

1. A balanced AC Newton--Raphson power-flow solver.
2. PV, PQ, and slack-bus handling.
3. An outer GFM controller with common-frequency P-f droop sharing.
4. A Q-V droop voltage loop with inverter capability limiting.
5. A Gym-compatible reinforcement-learning environment.
6. A small five-bus demonstration that runs with only NumPy.

Sign convention: positive P/Q is injection and positive load is consumption.
All electrical quantities are in per unit except frequency, which is in hertz.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # Keep the power-flow/controller usable even when Gym is not installed.
    import gymnasium as gym
except ImportError:  # PowerGym currently uses the older Gym package.
    try:
        import gym
    except ImportError:
        gym = None


_EnvBase = gym.Env if gym is not None else object


@dataclass
class PowerFlowResult:
    voltage: np.ndarray
    p_injection: np.ndarray
    q_injection: np.ndarray
    branch_p_from: np.ndarray
    branch_q_from: np.ndarray
    total_active_loss: float
    converged: bool
    iterations: int
    max_mismatch: float


class ACNetwork:
    """Balanced AC network with a from-scratch Newton--Raphson solver.

    Args:
        n_bus: Number of buses.
        branches: ``(from_bus, to_bus, resistance_pu, reactance_pu)`` rows.
        slack_bus: Angle-reference bus.
    """

    def __init__(
        self,
        n_bus: int,
        branches: Sequence[Tuple[int, int, float, float]],
        slack_bus: int = 0,
    ):
        self.n_bus = int(n_bus)
        self.branches = list(branches)
        self.slack_bus = int(slack_bus)
        if self.n_bus < 2 or not 0 <= self.slack_bus < self.n_bus:
            raise ValueError("invalid network dimensions or slack bus")

        self.ybus = np.zeros((self.n_bus, self.n_bus), dtype=np.complex128)
        for from_bus, to_bus, resistance, reactance in self.branches:
            if not (0 <= from_bus < self.n_bus and 0 <= to_bus < self.n_bus):
                raise ValueError("branch contains an invalid bus index")
            impedance = complex(resistance, reactance)
            if abs(impedance) == 0:
                raise ValueError("branch impedance cannot be zero")
            admittance = 1.0 / impedance
            self.ybus[from_bus, from_bus] += admittance
            self.ybus[to_bus, to_bus] += admittance
            self.ybus[from_bus, to_bus] -= admittance
            self.ybus[to_bus, from_bus] -= admittance

    def solve(
        self,
        p_spec: np.ndarray,
        q_spec: np.ndarray,
        pv_buses: Iterable[int],
        voltage_setpoints: np.ndarray,
        initial_voltage: Optional[np.ndarray] = None,
        tolerance: float = 1e-9,
        max_iterations: int = 40,
    ) -> PowerFlowResult:
        """Solve AC power flow.

        ``p_spec`` is specified at every non-slack bus. ``q_spec`` is used at
        PQ buses. Voltage magnitude is fixed at PV and slack buses.
        """
        p_spec = np.asarray(p_spec, dtype=float)
        q_spec = np.asarray(q_spec, dtype=float)
        voltage_setpoints = np.asarray(voltage_setpoints, dtype=float)
        if p_spec.shape != (self.n_bus,) or q_spec.shape != (self.n_bus,):
            raise ValueError("P and Q specifications must have one value per bus")

        pv = np.array(sorted(set(int(bus) for bus in pv_buses)), dtype=int)
        if self.slack_bus in pv:
            raise ValueError("slack bus must not also be listed as a PV bus")
        non_slack = np.array(
            [bus for bus in range(self.n_bus) if bus != self.slack_bus],
            dtype=int,
        )
        pq = np.array([bus for bus in non_slack if bus not in set(pv)], dtype=int)

        if initial_voltage is None:
            voltage_mag = np.ones(self.n_bus)
            voltage_angle = np.zeros(self.n_bus)
        else:
            initial_voltage = np.asarray(initial_voltage, dtype=np.complex128)
            voltage_mag = np.abs(initial_voltage).copy()
            voltage_angle = np.angle(initial_voltage).copy()

        voltage_mag[self.slack_bus] = voltage_setpoints[self.slack_bus]
        voltage_angle[self.slack_bus] = 0.0
        voltage_mag[pv] = voltage_setpoints[pv]

        converged = False
        max_mismatch = np.inf
        p_calc = np.zeros(self.n_bus)
        q_calc = np.zeros(self.n_bus)

        for iteration in range(1, max_iterations + 1):
            voltage = voltage_mag * np.exp(1j * voltage_angle)
            current = self.ybus @ voltage
            power = voltage * np.conj(current)
            p_calc = power.real
            q_calc = power.imag

            mismatch = np.concatenate(
                (p_spec[non_slack] - p_calc[non_slack],
                 q_spec[pq] - q_calc[pq])
            )
            max_mismatch = float(np.max(np.abs(mismatch))) if mismatch.size else 0.0
            if max_mismatch < tolerance:
                converged = True
                break

            jacobian = self._jacobian(
                voltage_mag, voltage_angle, p_calc, q_calc, non_slack, pq)
            try:
                correction = np.linalg.solve(jacobian, mismatch)
            except np.linalg.LinAlgError:
                break

            n_angle = len(non_slack)
            voltage_angle[non_slack] += correction[:n_angle]
            if len(pq):
                voltage_mag[pq] += correction[n_angle:]
                # Prevent Newton steps from crossing the nonphysical V=0 point.
                voltage_mag[pq] = np.clip(voltage_mag[pq], 0.2, 2.0)
            voltage_mag[self.slack_bus] = voltage_setpoints[self.slack_bus]
            voltage_mag[pv] = voltage_setpoints[pv]

        voltage = voltage_mag * np.exp(1j * voltage_angle)
        power = voltage * np.conj(self.ybus @ voltage)
        p_calc, q_calc = power.real, power.imag
        branch_p, branch_q, active_loss = self._branch_flows(voltage)
        return PowerFlowResult(
            voltage=voltage,
            p_injection=p_calc,
            q_injection=q_calc,
            branch_p_from=branch_p,
            branch_q_from=branch_q,
            total_active_loss=active_loss,
            converged=converged,
            iterations=iteration,
            max_mismatch=max_mismatch,
        )

    def _jacobian(
        self,
        voltage_mag: np.ndarray,
        voltage_angle: np.ndarray,
        p_calc: np.ndarray,
        q_calc: np.ndarray,
        non_slack: np.ndarray,
        pq: np.ndarray,
    ) -> np.ndarray:
        conductance, susceptance = self.ybus.real, self.ybus.imag
        n_angle, n_voltage = len(non_slack), len(pq)
        h = np.zeros((n_angle, n_angle))
        n = np.zeros((n_angle, n_voltage))
        m = np.zeros((n_voltage, n_angle))
        l = np.zeros((n_voltage, n_voltage))

        for row, bus_i in enumerate(non_slack):
            for col, bus_k in enumerate(non_slack):
                if bus_i == bus_k:
                    h[row, col] = (
                        -q_calc[bus_i]
                        - susceptance[bus_i, bus_i] * voltage_mag[bus_i] ** 2
                    )
                else:
                    angle = voltage_angle[bus_i] - voltage_angle[bus_k]
                    h[row, col] = voltage_mag[bus_i] * voltage_mag[bus_k] * (
                        conductance[bus_i, bus_k] * np.sin(angle)
                        - susceptance[bus_i, bus_k] * np.cos(angle)
                    )
            for col, bus_k in enumerate(pq):
                if bus_i == bus_k:
                    n[row, col] = (
                        p_calc[bus_i] / voltage_mag[bus_i]
                        + conductance[bus_i, bus_i] * voltage_mag[bus_i]
                    )
                else:
                    angle = voltage_angle[bus_i] - voltage_angle[bus_k]
                    n[row, col] = voltage_mag[bus_i] * (
                        conductance[bus_i, bus_k] * np.cos(angle)
                        + susceptance[bus_i, bus_k] * np.sin(angle)
                    )

        for row, bus_i in enumerate(pq):
            for col, bus_k in enumerate(non_slack):
                if bus_i == bus_k:
                    m[row, col] = (
                        p_calc[bus_i]
                        - conductance[bus_i, bus_i] * voltage_mag[bus_i] ** 2
                    )
                else:
                    angle = voltage_angle[bus_i] - voltage_angle[bus_k]
                    m[row, col] = -voltage_mag[bus_i] * voltage_mag[bus_k] * (
                        conductance[bus_i, bus_k] * np.cos(angle)
                        + susceptance[bus_i, bus_k] * np.sin(angle)
                    )
            for col, bus_k in enumerate(pq):
                if bus_i == bus_k:
                    l[row, col] = (
                        q_calc[bus_i] / voltage_mag[bus_i]
                        - susceptance[bus_i, bus_i] * voltage_mag[bus_i]
                    )
                else:
                    angle = voltage_angle[bus_i] - voltage_angle[bus_k]
                    l[row, col] = voltage_mag[bus_i] * (
                        conductance[bus_i, bus_k] * np.sin(angle)
                        - susceptance[bus_i, bus_k] * np.cos(angle)
                    )
        return np.block([[h, n], [m, l]])

    def _branch_flows(
        self, voltage: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        p_from, q_from = [], []
        total_loss = 0.0
        for from_bus, to_bus, resistance, reactance in self.branches:
            current = (voltage[from_bus] - voltage[to_bus]) / complex(
                resistance, reactance)
            sending_power = voltage[from_bus] * np.conj(current)
            p_from.append(sending_power.real)
            q_from.append(sending_power.imag)
            total_loss += resistance * abs(current) ** 2
        return np.asarray(p_from), np.asarray(q_from), float(total_loss)


@dataclass
class GFMUnit:
    bus: int
    p_max: float
    s_max: float
    p_reference: float
    voltage_reference: float = 1.0
    q_reference: float = 0.0
    p_droop_hz_per_pu: float = 0.5
    q_droop_v_per_pu: float = 0.05
    voltage_min: float = 0.90
    voltage_max: float = 1.10

    def __post_init__(self):
        if self.p_max <= 0 or self.s_max < self.p_max:
            raise ValueError("GFM rating must satisfy s_max >= p_max > 0")
        if self.p_droop_hz_per_pu <= 0 or self.q_droop_v_per_pu <= 0:
            raise ValueError("GFM droop coefficients must be positive")


@dataclass
class GFMControlResult:
    power_flow: PowerFlowResult
    frequency_hz: float
    gfm_p: np.ndarray
    gfm_q: np.ndarray
    voltage_commands: np.ndarray
    q_limited: np.ndarray
    saturated: np.ndarray
    converged: bool
    control_iterations: int
    frequency_residual: float


class GFMVoltageController:
    """Outer grid-forming control loop wrapped around AC power flow.

    The slack-bus GFM closes active-power balance. A common frequency is found
    such that its actual slack power matches its P-f droop command. Other GFM
    buses are PV buses unless their reactive capability is reached, in which
    case they switch to PQ operation at the Q limit. Voltage commands are
    updated by the Q-V droop loop after every AC power-flow solution.
    """

    def __init__(
        self,
        network: ACNetwork,
        units: Sequence[GFMUnit],
        nominal_frequency_hz: float = 60.0,
        frequency_bounds_hz: Tuple[float, float] = (57.0, 63.0),
        voltage_relaxation: float = 0.35,
        frequency_relaxation: float = 0.8,
        max_control_iterations: int = 40,
        frequency_tolerance: float = 1e-7,
        voltage_tolerance: float = 1e-7,
    ):
        self.network = network
        self.units = list(units)
        self.nominal_frequency_hz = float(nominal_frequency_hz)
        self.frequency_bounds_hz = frequency_bounds_hz
        self.voltage_relaxation = float(voltage_relaxation)
        self.frequency_relaxation = float(frequency_relaxation)
        self.max_control_iterations = int(max_control_iterations)
        self.frequency_tolerance = float(frequency_tolerance)
        self.voltage_tolerance = float(voltage_tolerance)

        buses = [unit.bus for unit in self.units]
        if len(buses) != len(set(buses)):
            raise ValueError("only one GFM unit per bus is supported")
        if self.network.slack_bus not in buses:
            raise ValueError("one GFM unit must be located at the slack bus")
        if any(not 0 <= bus < self.network.n_bus for bus in buses):
            raise ValueError("GFM unit contains an invalid bus")
        self.slack_unit = buses.index(self.network.slack_bus)
        self._last_voltage = np.ones(self.network.n_bus, dtype=np.complex128)
        self._last_frequency_hz = self.nominal_frequency_hz

    def solve(
        self,
        load_p: np.ndarray,
        load_q: np.ndarray,
        p_references: Optional[np.ndarray] = None,
        voltage_references: Optional[np.ndarray] = None,
    ) -> GFMControlResult:
        load_p = np.asarray(load_p, dtype=float)
        load_q = np.asarray(load_q, dtype=float)
        if load_p.shape != (self.network.n_bus,) or load_q.shape != load_p.shape:
            raise ValueError("load arrays must have one value per bus")
        if np.any(load_p < 0):
            raise ValueError("loads use a positive-consumption convention")

        p_ref = np.array(
            [unit.p_reference for unit in self.units]
            if p_references is None else p_references,
            dtype=float,
        )
        v_ref = np.array(
            [unit.voltage_reference for unit in self.units]
            if voltage_references is None else voltage_references,
            dtype=float,
        )
        if p_ref.shape != (len(self.units),) or v_ref.shape != p_ref.shape:
            raise ValueError("one P and V reference is required per GFM unit")

        frequency_deviation = self._last_frequency_hz - self.nominal_frequency_hz
        voltage_commands = v_ref.copy()
        q_limited = np.zeros(len(self.units), dtype=bool)
        q_limits = np.zeros(len(self.units))
        final_pf = None
        final_p = final_q = None
        final_frequency_residual = np.inf
        controller_converged = False

        for control_iteration in range(1, self.max_control_iterations + 1):
            limit_adjusted = False
            p_command = np.array([
                np.clip(
                    p_ref[index] - frequency_deviation /
                    unit.p_droop_hz_per_pu,
                    -unit.p_max,
                    unit.p_max,
                )
                for index, unit in enumerate(self.units)
            ])

            p_spec = -load_p.copy()
            q_spec = -load_q.copy()
            pv_buses: List[int] = []
            voltage_setpoints = np.ones(self.network.n_bus)

            for index, unit in enumerate(self.units):
                voltage_setpoints[unit.bus] = voltage_commands[index]
                if unit.bus != self.network.slack_bus:
                    p_spec[unit.bus] += p_command[index]
                    if q_limited[index]:
                        q_capability = np.sqrt(max(
                            0.0, unit.s_max ** 2 - p_command[index] ** 2))
                        adjusted_limit = float(np.clip(
                            q_limits[index], -q_capability, q_capability))
                        limit_adjusted = limit_adjusted or not np.isclose(
                            adjusted_limit, q_limits[index], atol=1e-10)
                        q_limits[index] = adjusted_limit
                        q_spec[unit.bus] += q_limits[index]
                    else:
                        pv_buses.append(unit.bus)

            final_pf = self.network.solve(
                p_spec,
                q_spec,
                pv_buses,
                voltage_setpoints,
                initial_voltage=self._last_voltage,
            )

            final_p = np.array([
                final_pf.p_injection[unit.bus] + load_p[unit.bus]
                for unit in self.units
            ])
            final_q = np.array([
                final_pf.q_injection[unit.bus] + load_q[unit.bus]
                for unit in self.units
            ])

            final_frequency_residual = (
                final_p[self.slack_unit] - p_command[self.slack_unit]
            )
            newly_limited = False
            for index, unit in enumerate(self.units):
                q_capability = np.sqrt(max(0.0, unit.s_max ** 2 - final_p[index] ** 2))
                if (
                    index != self.slack_unit
                    and not q_limited[index]
                    and abs(final_q[index]) > q_capability + 1e-8
                ):
                    q_limited[index] = True
                    q_limits[index] = np.clip(
                        final_q[index], -q_capability, q_capability)
                    newly_limited = True
                elif index != self.slack_unit and q_limited[index]:
                    adjusted_limit = float(np.clip(
                        q_limits[index], -q_capability, q_capability))
                    limit_adjusted = limit_adjusted or not np.isclose(
                        adjusted_limit, q_limits[index], atol=1e-10)
                    q_limits[index] = adjusted_limit

            target_voltage = np.array([
                np.clip(
                    v_ref[index] - unit.q_droop_v_per_pu *
                    (final_q[index] - unit.q_reference),
                    unit.voltage_min,
                    unit.voltage_max,
                )
                for index, unit in enumerate(self.units)
            ])
            next_voltage_commands = (
                (1.0 - self.voltage_relaxation) * voltage_commands
                + self.voltage_relaxation * target_voltage
            )
            voltage_change = float(np.max(np.abs(
                next_voltage_commands - voltage_commands)))

            if (
                final_pf.converged
                and abs(final_frequency_residual) < self.frequency_tolerance
                and voltage_change < self.voltage_tolerance
                and not newly_limited
                and not limit_adjusted
            ):
                controller_converged = True
                break

            # Approximate derivative of the slack droop mismatch with respect
            # to frequency. Exclude units already pinned at a P limit.
            droop_slope = 0.0
            for index, unit in enumerate(self.units):
                unconstrained = (
                    p_ref[index]
                    - frequency_deviation / unit.p_droop_hz_per_pu
                )
                if -unit.p_max < unconstrained < unit.p_max:
                    droop_slope += 1.0 / unit.p_droop_hz_per_pu
            droop_slope = max(droop_slope, 1e-9)
            frequency_deviation -= (
                self.frequency_relaxation * final_frequency_residual /
                droop_slope)
            frequency_deviation = float(np.clip(
                frequency_deviation,
                self.frequency_bounds_hz[0] - self.nominal_frequency_hz,
                self.frequency_bounds_hz[1] - self.nominal_frequency_hz,
            ))
            voltage_commands = next_voltage_commands
            self._last_voltage = final_pf.voltage

        frequency_hz = self.nominal_frequency_hz + frequency_deviation
        apparent_power = np.sqrt(final_p ** 2 + final_q ** 2)
        saturated = np.array([
            q_limited[index]
            or abs(final_p[index]) >= unit.p_max - 1e-8
            or apparent_power[index] >= unit.s_max - 1e-8
            for index, unit in enumerate(self.units)
        ])
        self._last_voltage = final_pf.voltage
        self._last_frequency_hz = frequency_hz
        return GFMControlResult(
            power_flow=final_pf,
            frequency_hz=frequency_hz,
            gfm_p=final_p,
            gfm_q=final_q,
            voltage_commands=voltage_commands,
            q_limited=q_limited,
            saturated=saturated,
            converged=controller_converged,
            control_iterations=control_iteration,
            frequency_residual=float(final_frequency_residual),
        )

    def reset(self):
        self._last_voltage = np.ones(self.network.n_bus, dtype=np.complex128)
        self._last_frequency_hz = self.nominal_frequency_hz


class GFMPowerFlowEnv(_EnvBase):
    """Gym-style environment backed by the custom AC/GFM solver.

    Action order is ``[P*_1, V*_1, P*_2, V*_2, ...]``. P references are in
    per unit on system base and V references are in per unit. The observation
    concatenates bus voltage magnitude, bus angle, normalized GFM P and Q,
    common frequency, and aggregate load scale.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        network: ACNetwork,
        controller: GFMVoltageController,
        base_load_p: np.ndarray,
        base_load_q: np.ndarray,
        horizon: int = 24,
        seed: int = 0,
    ):
        if gym is not None:
            super().__init__()
        self.network = network
        self.controller = controller
        self.base_load_p = np.asarray(base_load_p, dtype=float)
        self.base_load_q = np.asarray(base_load_q, dtype=float)
        self.horizon = int(horizon)
        self.rng = np.random.RandomState(seed)
        self.t = 0
        self.last_action = self.default_action()
        self.last_result: Optional[GFMControlResult] = None
        self.load_scale = 1.0

        if self.base_load_p.shape != (network.n_bus,):
            raise ValueError("base load must have one value per bus")

        action_low, action_high = [], []
        for unit in self.controller.units:
            action_low.extend([-unit.p_max, 0.95])
            action_high.extend([unit.p_max, 1.05])
        self._action_low = np.asarray(action_low, dtype=np.float32)
        self._action_high = np.asarray(action_high, dtype=np.float32)

        n_bus, n_gfm = network.n_bus, len(controller.units)
        observation_low = np.concatenate((
            np.full(n_bus, 0.0),
            np.full(n_bus, -np.pi),
            np.full(n_gfm, -1.5),
            np.full(n_gfm, -1.5),
            [0.90, 0.5],
        )).astype(np.float32)
        observation_high = np.concatenate((
            np.full(n_bus, 2.0),
            np.full(n_bus, np.pi),
            np.full(n_gfm, 1.5),
            np.full(n_gfm, 1.5),
            [1.10, 1.5],
        )).astype(np.float32)

        if gym is not None:
            self.action_space = gym.spaces.Box(
                self._action_low, self._action_high, dtype=np.float32)
            self.observation_space = gym.spaces.Box(
                observation_low, observation_high, dtype=np.float32)

    def default_action(self) -> np.ndarray:
        return np.asarray([
            value
            for unit in self.controller.units
            for value in (unit.p_reference, unit.voltage_reference)
        ], dtype=np.float32)

    def seed(self, seed: int):
        self.rng = np.random.RandomState(seed)
        if gym is not None:
            self.action_space.seed(seed)

    def reset(self, seed: Optional[int] = None, options=None):
        del options
        if seed is not None:
            self.seed(seed)
        self.t = 0
        self.load_scale = 1.0
        self.last_action = self.default_action()
        self.controller.reset()
        self.last_result = self._solve(self.last_action)
        return self._observation(self.last_result)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float)
        if action.shape != self._action_low.shape:
            raise ValueError("action must contain one [P*, V*] pair per GFM")
        action = np.clip(action, self._action_low, self._action_high)
        self.t += 1
        self.load_scale = float(
            1.0 + 0.12 * np.sin(2.0 * np.pi * self.t / self.horizon)
            + self.rng.normal(0.0, 0.005))
        self.last_result = self._solve(action)

        voltage_mag = np.abs(self.last_result.power_flow.voltage)
        voltage_violation = np.sum(
            np.maximum(0.95 - voltage_mag, 0.0)
            + np.maximum(voltage_mag - 1.05, 0.0))
        frequency_error = abs(
            self.last_result.frequency_hz
            - self.controller.nominal_frequency_hz)
        action_movement = np.linalg.norm(action - self.last_action)
        reward = -(
            100.0 * voltage_violation
            + 2.0 * frequency_error
            + self.last_result.power_flow.total_active_loss
            + 0.1 * np.sum(self.last_result.saturated)
            + 0.01 * action_movement)
        if not self.last_result.converged:
            reward -= 10.0
        self.last_action = action.astype(np.float32)
        done = self.t >= self.horizon
        info: Dict[str, object] = {
            "power_flow_converged": self.last_result.power_flow.converged,
            "gfm_control_converged": self.last_result.converged,
            "power_flow_iterations": self.last_result.power_flow.iterations,
            "control_iterations": self.last_result.control_iterations,
            "frequency_hz": self.last_result.frequency_hz,
            "frequency_residual_pu": self.last_result.frequency_residual,
            "active_loss_pu": self.last_result.power_flow.total_active_loss,
            "gfm_p_pu": self.last_result.gfm_p.copy(),
            "gfm_q_pu": self.last_result.gfm_q.copy(),
            "q_limited": self.last_result.q_limited.copy(),
            "saturated": self.last_result.saturated.copy(),
            "load_scale": self.load_scale,
        }
        return self._observation(self.last_result), float(reward), done, info

    def _solve(self, action: np.ndarray) -> GFMControlResult:
        p_references = action[0::2]
        voltage_references = action[1::2]
        return self.controller.solve(
            self.base_load_p * self.load_scale,
            self.base_load_q * self.load_scale,
            p_references,
            voltage_references,
        )

    def _observation(self, result: GFMControlResult) -> np.ndarray:
        p_normalized = np.array([
            result.gfm_p[index] / unit.p_max
            for index, unit in enumerate(self.controller.units)
        ])
        q_normalized = np.array([
            result.gfm_q[index] / unit.s_max
            for index, unit in enumerate(self.controller.units)
        ])
        return np.concatenate((
            np.abs(result.power_flow.voltage),
            np.angle(result.power_flow.voltage),
            p_normalized,
            q_normalized,
            [result.frequency_hz / self.controller.nominal_frequency_hz],
            [self.load_scale],
        )).astype(np.float32)


def make_demo_env(seed: int = 0) -> GFMPowerFlowEnv:
    """Return a complete five-bus example with two GFM units."""
    network = ACNetwork(
        n_bus=5,
        branches=[
            (0, 1, 0.010, 0.040),
            (1, 2, 0.012, 0.045),
            (1, 3, 0.015, 0.050),
            (3, 4, 0.010, 0.035),
        ],
        slack_bus=0,
    )
    units = [
        GFMUnit(
            bus=0,
            p_max=1.50,
            s_max=1.80,
            p_reference=0.80,
            p_droop_hz_per_pu=0.50,
            q_droop_v_per_pu=0.04,
        ),
        GFMUnit(
            bus=3,
            p_max=0.80,
            s_max=1.00,
            p_reference=0.50,
            p_droop_hz_per_pu=0.50,
            q_droop_v_per_pu=0.05,
        ),
    ]
    controller = GFMVoltageController(network, units)
    return GFMPowerFlowEnv(
        network,
        controller,
        base_load_p=np.array([0.0, 0.45, 0.35, 0.25, 0.20]),
        base_load_q=np.array([0.0, 0.18, 0.14, 0.10, 0.08]),
        horizon=24,
        seed=seed,
    )


if __name__ == "__main__":
    environment = make_demo_env(seed=7)
    observation = environment.reset()
    print("Initial observation dimension:", observation.shape[0])
    print("Initial voltage magnitudes:", np.round(
        np.abs(environment.last_result.power_flow.voltage), 5))
    for step_index in range(6):
        next_observation, reward, done, step_info = environment.step(
            environment.default_action())
        print(
            "step={:02d} reward={:8.4f} f={:.5f}Hz loss={:.6f} "
            "V=[{:.4f}, {:.4f}] converged={}".format(
                step_index + 1,
                reward,
                step_info["frequency_hz"],
                step_info["active_loss_pu"],
                np.min(next_observation[:environment.network.n_bus]),
                np.max(next_observation[:environment.network.n_bus]),
                step_info["gfm_control_converged"],
            )
        )
        if done:
            break
