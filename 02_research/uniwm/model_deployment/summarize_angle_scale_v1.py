"""Report all signed angles and full denominators, without selecting outcomes."""
import argparse
from pathlib import Path
from collections import defaultdict
from suite_common import read,digest,atomic
from angle_scale_common import root,ANGLES,NEW_ANGLES
from habitat_flow_common import direction_counts


def mean(xs):
    xs=[x for x in xs if x is not None];return sum(xs)/len(xs) if xs else None


def aggregate(rows):
    counts=direction_counts([r.get('action_direction',dict(label='unknown')) for r in rows])
    return dict(direction=counts,verified_direction_pass_rate=counts['correct']/len(rows) if rows else None,
        acceptance_not_passed_including_unknown=len(rows)-counts['correct'],unknown_is_observed_turn_error=False,
        mean_effect_epe_px=mean([r.get('action_effect',{}).get('mean_epe_px') for r in rows if r.get('reference_gate_passed')]),
        mean_gain=mean([r.get('action_effect',{}).get('gain_projection') for r in rows if r.get('reference_gate_passed')]),
        mean_action_zero_lpips=mean([r.get('action_vs_zero_lpips') for r in rows]),
        paired_zero_identical=sum(r.get('image_identical_to_paired_zero',False) for r in rows))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);a=ap.parse_args();out=root(a.study)
    result=dict(status='passed',protocol_sha256=digest(out/'protocol.json'),models={},world_model_updates=0,
        new_predictions=432,reused_predictions=756,
        limits=['±10/15deg exceed ±7deg Habitat adaptation support; base pretraining support unknown',
        'existing12 positions previously inspected; this is exploratory follow-up, not a fresh independent confirmation',
        '2 scenes; seeds and actions are repeated measurements',
        'pixel gains are not physical angle accuracy; unknown is measurement limitation',
        'larger-response evidence would not establish an in-distribution small-angle-specific cause'])
    lines=['# 小角度响应与较大角度外推对照','',
        '±3°/5°/7°属于本轮适配范围；新增±10°/15°超出适配范围，基座完整预训练支持未知。',
        '沿用两个场景的12位置、原3个配对seed、原权重；新增432张预测，复用756张旧图，训练更新0。',
        '所有11个真实角度共同决定本扩展的固定掩码，旧冻结主表不改；本表的小角度为同掩码重评分。','',
        '| 模型 | 角度绝对值 | 同向 | 反向 | 近零 | 未知 | 分母 |','|---|---:|---:|---:|---:|---:|---:|']
    new_count=0
    for model in ['nwm','raenwm','uniwm']:
        folder=out/'confirmation'/model;done=read(folder/'complete.json');scored=read(folder/'analysis/complete.json')
        assert done['records']==396 and scored['records']==396
        rows=read(folder/'analysis/rows.json');assert len(rows)==396
        nonzero=[r for r in rows if r['angle']];large=[r for r in nonzero if abs(r['angle']) in [10,15]]
        assert len(large)==144;new_count+=len(large)
        block=dict(small=aggregate([r for r in nonzero if abs(r['angle'])<=7]),large=aggregate(large),
            by_magnitude={},by_signed_angle={},by_scene={},by_position={},original_small_frozen=read(a.study/'v2/paper_confirmation_v1/confirmation'/model/'analysis/summary.json')['overall'],
            analysis_complete_sha256=digest(folder/'analysis/complete.json'))
        for deg in [3,5,7,10,15]:
            v=aggregate([r for r in nonzero if abs(r['angle'])==deg]);block['by_magnitude'][str(deg)]=v;c=v['direction']
            lines.append(f'| {model} | {deg} | {c["correct"]} | {c["reverse"]} | {c["near_zero"]} | {c["unknown"]} | {c["total"]} |')
        for angle in ANGLES[1:]:block['by_signed_angle'][str(angle)]=aggregate([r for r in nonzero if r['angle']==angle])
        for field,key in [('scene_id','by_scene'),('state_id','by_position')]:
            for v in sorted({r[field] for r in nonzero}):
                subset=[r for r in nonzero if r[field]==v]
                block[key][v]=dict(small=aggregate([r for r in subset if abs(r['angle'])<=7]),large=aggregate([r for r in subset if abs(r['angle'])>7]))
        result['models'][model]=block
    assert new_count==432
    lines+=['','未知计入“验收不通过”的补充口径，但不当作已观察到的转向错误；不据此排名真实转向能力。',
        '较大角度若出现稳定响应，可支持幅度相关的行为差异；若同样近零，则不支持只在小角度失效；若覆盖不足，则该模型仍无法判断。',
        '无论哪种结果，超出适配范围的对照都不能证明训练分布内的小角度特有机制。']
    atomic(out/'summary.json',result);(out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    atomic(out/'complete.json',dict(status='passed',new_predictions=432,summary_sha256=digest(out/'summary.json'),world_model_updates=0))
    print('ANGLE_SCALE_COMPLETE:432 new predictions, all scores retained',flush=True)


if __name__=='__main__':main()
