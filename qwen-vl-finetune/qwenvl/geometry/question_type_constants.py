MCA_QUESTION_TYPES = (
    'appearance_order',
    'relative_count',
    'relative_direction_object',
    'relative_distance_object',
    'relative_size_object',
    'route_planning',
)

NA_QUESTION_TYPES = (
    'absolute_count',
    'absolute_direction_object',
    'absolute_distance_object',
    'absolute_size_object',
    'absolute_size_room',
    'object_abs_distance',
)

MCA_LOW_LEVEL_TYPES = (
    'distance_infer',
)

L1_TYPES = (
    'distance_to_camera',
    'distance_prediction',
)

L2_TYPES = (
    'position_matching',
    'spatial_imagination_3d',
)

MRA_COUNTING_TYPES = (
    'counting',
)

CAPTION_TYPES = (
    'novel_view_captioning',
)

VG_LLM_CAPTION_LOSS_TYPES = (
    'vg_llm_qa_scan2cap',
)

VG_LLM_BBOX_FRAME_TYPES = (
    'vg_llm_qa_scanrefer',
)

VG_LLM_BBOX_DET_TYPES = (
    'vg_llm_qa_scannet_det',
)

QTYPE_ID_UNKNOWN = -1

ALL_MCA_TYPES = MCA_QUESTION_TYPES + MCA_LOW_LEVEL_TYPES

_ALL_TYPES = tuple(sorted(
    set(MCA_QUESTION_TYPES)
    | set(NA_QUESTION_TYPES)
    | set(MCA_LOW_LEVEL_TYPES)
    | set(L1_TYPES)
    | set(L2_TYPES)
    | set(MRA_COUNTING_TYPES)
    | set(CAPTION_TYPES)
    | set(VG_LLM_CAPTION_LOSS_TYPES)
    | set(VG_LLM_BBOX_FRAME_TYPES)
    | set(VG_LLM_BBOX_DET_TYPES)
))
QUESTION_TYPE_TO_ID = {name: i for i, name in enumerate(_ALL_TYPES)}
ID_TO_QUESTION_TYPE = {i: name for name, i in QUESTION_TYPE_TO_ID.items()}

MCA_TYPE_IDS = frozenset(QUESTION_TYPE_TO_ID[t] for t in ALL_MCA_TYPES)
NA_TYPE_IDS = frozenset(QUESTION_TYPE_TO_ID[t] for t in NA_QUESTION_TYPES)
L1_TYPE_IDS = frozenset(QUESTION_TYPE_TO_ID[t] for t in L1_TYPES)
L2_TYPE_IDS = frozenset(QUESTION_TYPE_TO_ID[t] for t in L2_TYPES)
MRA_COUNTING_TYPE_IDS = frozenset(QUESTION_TYPE_TO_ID[t] for t in MRA_COUNTING_TYPES)
CAPTION_TYPE_IDS = frozenset(QUESTION_TYPE_TO_ID[t] for t in CAPTION_TYPES)
VG_LLM_CAPTION_LOSS_TYPE_IDS = frozenset(QUESTION_TYPE_TO_ID[t] for t in VG_LLM_CAPTION_LOSS_TYPES)
VG_LLM_BBOX_FRAME_TYPE_IDS = frozenset(QUESTION_TYPE_TO_ID[t] for t in VG_LLM_BBOX_FRAME_TYPES)
VG_LLM_BBOX_DET_TYPE_IDS = frozenset(QUESTION_TYPE_TO_ID[t] for t in VG_LLM_BBOX_DET_TYPES)

ALL_METRIC_TYPE_IDS = (
    MCA_TYPE_IDS | NA_TYPE_IDS | L1_TYPE_IDS | L2_TYPE_IDS | MRA_COUNTING_TYPE_IDS
    | VG_LLM_CAPTION_LOSS_TYPE_IDS | VG_LLM_BBOX_FRAME_TYPE_IDS | VG_LLM_BBOX_DET_TYPE_IDS
)


def question_type_to_id(qt):
    if qt is None:
        return QTYPE_ID_UNKNOWN
    return QUESTION_TYPE_TO_ID.get(qt, QTYPE_ID_UNKNOWN)
