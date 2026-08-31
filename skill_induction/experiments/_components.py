"""Component factory: resolves agent and benchmark names to classes, so `--agent` and `--benchmark` are the only switches flipped between baselines."""

from __future__ import annotations

from typing import Any


def get_components(agent: str) -> dict:
    """Resolve agent name to {agent_cls, memory_factory or None}. The awm variants add doc-memory skill induction, the asi variants code-memory; plain linear/subtask run without memory."""
    from skill_induction.agents.linear import LinearAgent
    from skill_induction.agents.awm import LinearAWM
    from skill_induction.agents.asi import LinearASI
    from skill_induction.agents.subtask import SubtaskAgent
    from skill_induction.agents.subtask_awm import SubtaskAWM
    from skill_induction.agents.subtask_asi import SubtaskASI

    REGISTRY: dict[str, dict[str, Any]] = {
        "linear":       {"agent_cls": LinearAgent,  "memory_factory": None},
        "linear_awm":   {"agent_cls": LinearAWM,    "memory_factory": _awm_factory},
        "linear_asi":   {"agent_cls": LinearASI,    "memory_factory": _asi_factory},
        "subtask":      {"agent_cls": SubtaskAgent, "memory_factory": None},
        "subtask_awm":  {"agent_cls": SubtaskAWM,   "memory_factory": _awm_factory},
        "subtask_asi":  {"agent_cls": SubtaskASI,   "memory_factory": _asi_factory},
    }
    if agent not in REGISTRY:
        raise ValueError(
            f"Unknown agent {agent!r}. Choose one of: {sorted(REGISTRY)}.")
    return REGISTRY[agent]


def get_world_cls(benchmark: str):
    """Resolve benchmark name to a World class."""
    if benchmark == "appworld":
        from skill_induction.envs.appworld_env import RemoteAppWorld
        return RemoteAppWorld
    if benchmark == "officebench":
        from skill_induction.envs.officebench_env import OfficeBenchWorld
        return OfficeBenchWorld
    if benchmark == "kramabench":
        from skill_induction.envs.kramabench_env import KramaBenchWorld
        return KramaBenchWorld
    raise ValueError(f"Unknown benchmark: {benchmark}")


def get_server_cls(benchmark: str):
    """Resolve benchmark to a context manager that yields a server URL."""
    if benchmark == "appworld":
        from skill_induction.envs.appworld_env import AppWorldServer
        return AppWorldServer
    if benchmark == "officebench":
        from skill_induction.envs.officebench_env import OfficeBenchNoopServer
        return OfficeBenchNoopServer
    if benchmark == "kramabench":
        from skill_induction.envs.kramabench_env import KramaBenchNoopServer
        return KramaBenchNoopServer
    raise NotImplementedError(f"server class not wired for {benchmark}")


def _awm_factory(model):
    from skill_induction.memory.awm_memory import AWMMemory
    return AWMMemory(model=model)


def _asi_factory(model):
    from skill_induction.memory.asi_memory import ASIMemory
    return ASIMemory(model=model)


AGENT_NAMES = (
    "linear", "linear_awm", "linear_asi",
    "subtask", "subtask_awm", "subtask_asi",
)
