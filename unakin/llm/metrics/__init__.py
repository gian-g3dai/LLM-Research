from .base import BaseMetric, MetricValue, MetricType

from .perplexity import PerplexityMetric
from .utils import *

# NOTE: this is a mapping from the name of the metric to the class
METRIC_NAME_MAP = {
  "perplexity": PerplexityMetric,
}