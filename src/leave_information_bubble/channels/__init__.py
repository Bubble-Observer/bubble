"""Versioned, platform-neutral information channel contracts."""

from .acquisition import (
    StratifiedCardReservoir,
    build_retrieval_contract_report,
    canonicalize_public_url,
    cluster_origin_candidates,
    detect_public_text_language,
)
from .base import ChannelAdapter
from .bilibili import (
    BilibiliChannelAdapter,
    BilibiliCommentSampling,
    BilibiliCommentSort,
)
from .hupu import HupuChannelAdapter
from .models import (
    AcquisitionEntryKind,
    AcquisitionOutcome,
    CapabilityDescriptor,
    ChannelCapabilityRole,
    ChannelHealth,
    ChannelHealthStatus,
    DiscoveryBatch,
    HydrationDepth,
    HydrationRequest,
    IndependenceStatus,
    ObservationBatch,
    QuerySemantics,
    RetrievalBatch,
    RetrievalContractReport,
    RetrievalHit,
    ScanRequest,
    SourceOccurrence,
    TimeFilterPrecision,
)
from .nga import NgaChannelAdapter
from .public_web import PublicWebChannelAdapter
from .replay import ReplayChannelAdapter

__all__ = [
    "AcquisitionEntryKind",
    "AcquisitionOutcome",
    "ChannelAdapter",
    "BilibiliChannelAdapter",
    "BilibiliCommentSampling",
    "BilibiliCommentSort",
    "CapabilityDescriptor",
    "ChannelCapabilityRole",
    "ChannelHealth",
    "ChannelHealthStatus",
    "DiscoveryBatch",
    "HydrationDepth",
    "HydrationRequest",
    "HupuChannelAdapter",
    "IndependenceStatus",
    "NgaChannelAdapter",
    "ObservationBatch",
    "PublicWebChannelAdapter",
    "QuerySemantics",
    "ReplayChannelAdapter",
    "RetrievalBatch",
    "RetrievalContractReport",
    "RetrievalHit",
    "ScanRequest",
    "SourceOccurrence",
    "StratifiedCardReservoir",
    "TimeFilterPrecision",
    "build_retrieval_contract_report",
    "canonicalize_public_url",
    "cluster_origin_candidates",
    "detect_public_text_language",
]
