from dataclasses import dataclass, field
from typing import List, Optional, Union

from steerability.rewards import steerability_reward_wrapper

@dataclass
class SteerabilityProbeConfig:
    steerability_probe: str = field(
        default="./data/steerbench_converted.csv",
        metadata={"help": "Path to steerability probe (CSV file)."}
    )

    n_source_texts: int = field(
        default=1,
        metadata={"help": "Number of source texts to sample for training data."}
    )
    
    apply_conversational_format: bool = field(
        default=True,
        metadata={"help": "Whether to apply basic conversational request template (OpenAI style), e.g., `{'role': 'user', 'content': [PROMPT]}`"}
    )

    instructions_per_text: int = field(
        default=8,
        metadata={"help": "Number of instructions to sample per source text in steerability probe."}
    )

    probe_sampling_seed: int = field(
        default=137, 
        metadata={"help": "Random seed for subsampling steerability probe for training data preparation."}
    )

    test_source_sampling_seed: int = field(
        default=137,
        metadata={"help": "Random seed for selecting test source texts."}
    )
    test_inst_sampling_seed: int = field(
        default=137,
        metadata={"help": "Random seed for selecting test instructions."}
    )
    train_inst_sampling_seed: int = field(
        default=137,
        metadata={"help": "Random seed for selecting train source texts."}
    )
    train_source_sampling_seed: int = field(
        default=137,
        metadata={"help": "Random seed for selecting train instructions."}
    )

    source_text_id_col: str = field(
        default="original_index",
        metadata={"help": "Pandas DataFrame column name corresponding to a source text ID (an index, or the string itself)."}
    )
    
    source_text_col: str = field(
        default="text",
        metadata={"help": "Pandas DataFrame column containing the actual source texts."}
    )
    prompt_strategy: str = field(
        default="direct",
        metadata={"help": "Prompt strategy (see `instruction_generatory.py`) for generating instructions from goalspace vectors."}
    )
    num_train_prompts_for_eval: int = field(
        default=6,
        metadata={"help": "Number of training instructions used for evaluation between epochs. Drawn from training probe."}
    )
    num_test_prompts_for_eval: int = field(
        default=0,
        metadata={"help": "Number of new instructions used for evaluation between epochs. Drawn from seed set but never seen by model."}
    )
    insts_per_probe_source_text: Optional[int] = field(
        default=None,
        metadata={"help": "Number of instructions per probe to sample."}
    )
    canary_file: str = field(
        default=None,
        metadata={"help": "File containing lists of strings for evaluating completions only."}
    )
    canary_goal_magnitude: float = field(
        default=0.3,
        metadata={"help": "Magnitude of dummy goals given for manipulating canary texts."}
    )
    clip_min: float = field(
        default=0.0,
        metadata={"help": "Minimum sample weight for clipping."}
    )
    clip_max: float = field(
        default=float('inf'),
        metadata={"help": "Maximum sample weight for clipping."}
    )
    cross_probes: bool = field(
        default=True,
        metadata={"help": "Whether to generate cross-probes."}
    )
    train_probe_path: Optional[str] = field(
        default=None,
        metadata={"help": "Load a training dataset from another run. Useful for replications."}
    )
    test_probe_path: Optional[str] = field(
        default=None,
        metadata={"help": "Load a test probe from another run."}
    )

@dataclass
class GoalspaceServerConfig:
    port: int = field(
        default=12121,
        metadata={"help": "Port number for pinging goalspace-mapping server."}
    )
    startup_timeout: int = field(
        default = 300,
        metadata={"help": "Maximum time to wait (in seconds) for server to start up."}
    )
@dataclass
class RewardConfig:
    rewards: List[str] = field(
        default_factory=lambda: [steerability_reward_wrapper],
        metadata={"help": "List of rewards to optimize for. See rewards.REGISTRY."}
    )
    eval_rewards: List[str] = field(
        default_factory=list,
        metadata={"help": "List of rewards to evaluate during training only, but not optimize. See rewards.REGISTRY for a full list."}
    )
    steering_goals: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of goals to run. Ping endpoint /goals on goalspace server to get a list."}
    )
    normalize_ortho: bool = field(
        default=False,
        metadata={"help": "Whether to normalize the orthogonality penalty by magnitude moved in goal-space."}
    )
    normalize_miscal: bool = field(
        default=False,
        metadata={"help": "Whether to normalize the miscalibration penalty by the length of requested movement in goal-space."}
    )
    rescale_norm: bool = field(
        default=False,
        metadata={"help": "Whether to rescale the L2 norm metric to be in [0, 1]. By default, the range is [-sqrt(d), 0] or [-d, 0] depending on whether `squared=True`."}
    )
    square_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to optimize squared L2 in goalspace as a reward (instead of non-squared L2)."}
    )
    good_enough_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "Clip the goal dimension error below this threshold, such that `good_enough_threshold` is the max possible reward/min possible error in a goal dimension."}
    )
    too_bad_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "Clip the goal dimension error above this threshold to this value, such that `too_bad_threshold` is the min. possible reward/max. possible error in a goal dimension."}
    )
    decile: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to shape rewards by discretization into buckets of size 0.1."}
    )