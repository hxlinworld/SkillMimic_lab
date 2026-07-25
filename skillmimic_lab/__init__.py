"""Isaac Lab task registration for SkillMimic."""

import gymnasium as gym


TASK_ID = "SkillMimic-BallPlay-Direct-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="skillmimic_lab.env.tasks.skillmimic:SkillMimicBallPlayEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "skillmimic_lab.env.tasks.skillmimic:SkillMimicBallPlayEnvCfg",
            "rl_games_cfg_entry_point": "skillmimic_lab.agents:rl_games_ppo_cfg.yaml",
        },
    )

_HLC_TASKS = {
    "SkillMimic-Circling-Direct-v0": "skillmimic_lab.env.tasks.hrl_circling:SkillMimicCirclingEnvCfg",
    "SkillMimic-Heading-Direct-v0": "skillmimic_lab.env.tasks.hrl_heading_easy:SkillMimicHeadingEnvCfg",
    "SkillMimic-Throwing-Direct-v0": "skillmimic_lab.env.tasks.hrl_throwing:SkillMimicThrowingEnvCfg",
    "SkillMimic-Scoring-Direct-v0": "skillmimic_lab.env.tasks.hrl_scoring_layup:SkillMimicScoringEnvCfg",
}
for task_id, cfg_entry_point in _HLC_TASKS.items():
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point="skillmimic_lab.env.tasks.hrl_base:SkillMimicHLCEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": cfg_entry_point,
                "rl_games_cfg_entry_point": "skillmimic_lab.agents:rl_games_hlc_ppo_cfg.yaml",
            },
        )


__all__ = ["TASK_ID"]
