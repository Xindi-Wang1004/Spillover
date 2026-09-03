"""genome-ml-reportcard: estimand-matched audit for group-assigned genome ML labels."""

__version__ = "0.1.1"

SCHEMA_COLUMNS = (
    "sequence_id",  # accession / genome id
    "label",        # numeric or binary target
    "group",        # label-assignment unit (Layer A)
)
OPTIONAL_COLUMNS = (
    "block",        # deployment / blocking unit (Layer B; defaults to group)
    "taxonomy",
    "fasta_path",
    "split",
)
