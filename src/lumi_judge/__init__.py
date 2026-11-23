"""LUMI Standalone Partial-Screen Photosensitive Epilepsy (PSE) Detection & Mask Core Package (`vdcoreps`)."""

from __future__ import annotations

from .config import (
    ALL_HAZARD_LABELS,
    FPSWindowSpec,
    IntervalRecord,
    LABEL_INTERSECTION,
    LABEL_LUM_EXTENDED,
    LABEL_LUM_FLASH,
    LABEL_RED_EXTENDED,
    LABEL_RED_FLASH,
    LABEL_SAFE,
    LABEL_SCENE_TRANSITION,
    LABEL_STATIC_PATTERN,
    PSEConfig,
    PackedHazardBuffers,
    PartialMaskConfig,
    PartialTemporalScheduleMode,
    PartialVideoAnalysisResult,
    TransitionEvent,
    VideoAnalysisResult,
    ViewingConfig,
    ViewingGeometry,
    ViewingProfile,
    mask_to_spans,
)
from .metrics import (
    NomuraTemporalFilter,
    PixelDiagnosticTracker,
    compute_ppr_and_calibrated_series,
    compute_vfm_weibull_series,
)
from .photometry import PhotometricConverter, resolve_device
from .caption_ops import (
    CachedCaptionSprite,
    DEFAULT_CAPTION_HEIGHT_RATIO,
    DEFAULT_CAPTION_JOINER,
    DEFAULT_CAPTION_PREFIX,
    DEFAULT_CAPTION_SCALE,
    build_caption_sprite,
    build_frame_caption_schedule,
    format_frame_hazard_caption,
    overlay_hazard_caption_bgr,
)
from .mask_ops import (
    apply_guided_borderless_batch,
    apply_outward_spatial_feather_batch,
    apply_partial_b2_dimming_bgr,
    apply_partial_linear_dimming_bgr,
    build_mode_weight_masks,
    build_red_mode_weight_masks,
    build_soft_weight_masks_u8,
    clean_speckles_packed,
    compute_4metrics_union_from_result,
    pack_bool_mask_2d,
    temporal_bridge_spans_packed,
    temporal_sliding_or_packed,
    unpack_bool_masks_3d,
    upsample_weight_mask,
)
from .pipeline import (
    PartialDetectorCache,
    analyze_frame_stream,
    analyze_frames,
    analyze_video,
    evaluate_video_pse_risk,
)
from .spatial import StaticPatternDetector
from .spatial_mask import (
    PartialStaticPatternDetector,
    extract_hysteresis_extrema_1d,
    filter_static_pattern_runs,
    find_regular_open_runs_1d,
    morphological_close_2d,
    separable_max_min_2d,
)
from .temporal import TemporalTransitionEngine
from .temporal_mask import (
    PartialTemporalTransitionEngine,
    build_flash_boolean_mask,
    build_iris_extended_mask,
    merge_ultrafast_transitions,
    trailing_1s_event_stats,
)

# Alias matching `DetectorCache` naming convention for downstream convenience
DetectorCache = PartialDetectorCache

__all__ = [
    # Labels & constants
    "LABEL_LUM_FLASH",
    "LABEL_INTERSECTION",
    "LABEL_SCENE_TRANSITION",
    "LABEL_RED_FLASH",
    "LABEL_LUM_EXTENDED",
    "LABEL_RED_EXTENDED",
    "LABEL_STATIC_PATTERN",
    "LABEL_SAFE",
    "ALL_HAZARD_LABELS",
    "ViewingProfile",
    "PartialTemporalScheduleMode",
    # Config & result types
    "ViewingGeometry",
    "ViewingConfig",
    "PSEConfig",
    "FPSWindowSpec",
    "TransitionEvent",
    "IntervalRecord",
    "VideoAnalysisResult",
    "PartialMaskConfig",
    "PackedHazardBuffers",
    "PartialVideoAnalysisResult",
    # Photometry & device
    "resolve_device",
    "PhotometricConverter",
    # Temporal & spatial localization engines
    "TemporalTransitionEngine",
    "PartialTemporalTransitionEngine",
    "merge_ultrafast_transitions",
    "build_flash_boolean_mask",
    "trailing_1s_event_stats",
    "build_iris_extended_mask",
    "StaticPatternDetector",
    "PartialStaticPatternDetector",
    "find_regular_open_runs_1d",
    "morphological_close_2d",
    "extract_hysteresis_extrema_1d",
    "separable_max_min_2d",
    "filter_static_pattern_runs",
    # Continuous metrics
    "PixelDiagnosticTracker",
    "NomuraTemporalFilter",
    "compute_vfm_weibull_series",
    "compute_ppr_and_calibrated_series",
    # Spatiotemporal mask operations & Formula A helpers
    "pack_bool_mask_2d",
    "unpack_bool_masks_3d",
    "temporal_sliding_or_packed",
    "temporal_bridge_spans_packed",
    "clean_speckles_packed",
    "apply_guided_borderless_batch",
    "apply_outward_spatial_feather_batch",
    "build_soft_weight_masks_u8",
    "compute_4metrics_union_from_result",
    "build_mode_weight_masks",
    "build_red_mode_weight_masks",
    "upsample_weight_mask",
    "apply_partial_linear_dimming_bgr",
    "apply_partial_b2_dimming_bgr",
    # Hazard caption scheduling & overlay helpers
    "DEFAULT_CAPTION_PREFIX",
    "DEFAULT_CAPTION_JOINER",
    "DEFAULT_CAPTION_HEIGHT_RATIO",
    "DEFAULT_CAPTION_SCALE",
    "CachedCaptionSprite",
    "format_frame_hazard_caption",
    "build_frame_caption_schedule",
    "build_caption_sprite",
    "overlay_hazard_caption_bgr",
    # Pipeline entry points
    "DetectorCache",
    "PartialDetectorCache",
    "mask_to_spans",
    "analyze_frame_stream",
    "analyze_frames",
    "analyze_video",
    "evaluate_video_pse_risk",
]
