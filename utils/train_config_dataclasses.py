from dataclasses import dataclass, field

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

    instructions_per_text: int = field(
        default=8,
        metadata={"help": "Number of instructions to sample per source text in steerability probe."}
    )

    probe_sampling_seed: int = field(
        default=42, 
        metadata={"help": "Random seed for subsampling steerability probe for training data preparation."}
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