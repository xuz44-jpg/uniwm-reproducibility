"""Require every scoped result before declaring the inference stage complete."""
import argparse
from pathlib import Path
from suite_common import read,digest,atomic
from context_controls_common import root


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);a=ap.parse_args();out=root(a.study)
    report=dict(status='passed',native={},habitat={},conditions=0,new_predictions=0,reused_predictions=0,
        world_model_updates=0,scope='exploratory input and initialization controls; stage1, not population benchmark')
    lines=['# 输入、初始化与适配对照：第一阶段','',
        '0°及左右5°，12个已有位置、原repeat0；未训练、未换点，未按结果调阈值。',
        '这是已分析场景的探索性输入对照，不能当新的独立确认。','',
        '## 原生实际动作正对照','',
        '| 模型 | 权重 | 条件 | LPIPS至真实未来 | 复制当前帧LPIPS |','|---|---|---:|---:|---:|']
    for cohort in ['native','habitat']:
        for model in ['nwm','raenwm','uniwm']:
            run=out/cohort/model;done=read(run/'complete.json');ad=read(run/'analysis/complete.json')
            assert done['status']==ad['status']=='passed'
            summary=read(run/'analysis/summary.json');assert digest(run/'analysis/summary.json')==ad['summary_sha256']
            report[cohort][model]=summary
            report['conditions']+=done['records'];report['new_predictions']+=done['new'];report['reused_predictions']+=done['reused']
    for model,s in report['native'].items():
        for name,b in s['by_arm'].items():
            lines.append(f'| {model} | {name} | {b["records"]} | {b["lpips_to_actual_future"]:.5f} | {b["copy_current_lpips_to_future"]:.5f} |')
    lines+=['','## Habitat方向完整分母','',
        '| 模型 | 权重/输入 | 同向/反向/近零/未知 | Z+U | 未通过验收R+Z+U | 增益 |',
        '|---|---|---|---:|---:|---:|']
    for model,s in report['habitat'].items():
        for name,b in s['by_arm'].items():
            d=b['direction'];gain=b['gain'];g='NA' if gain is None else f'{gain:.5f}'
            counts='/'.join(str(d[k]) for k in ['correct','reverse','near_zero','unknown'])
            lines.append(f'| {model} | {name} | {counts} / {d["total"]} | {d["pooled_near_zero_unknown"]} | {d["acceptance_not_passed"]} | {g} |')
    assert (report['conditions'],report['new_predictions'],report['reused_predictions'])==(984,816,168)
    lines+=['','## 解释边界','',
        '- UniWM为我们的1X-200复现及其Habitat适配，不代表官方发布导航checkpoint。',
        '- 更换goal改变训练条件支持度；不能由输出变化认定动作响应恢复或唯一训练根因。',
        '- Z+U合并是汇总视图；未知不是实测零运动，验收未通过包含测量无法判定。',
        '- 一seed、12位置不等于12独立房屋；ReCon仅2源轨迹，物理时间对应仍有限。',
        '- 当前训练数据量与初始化差异仍未由本轮推理解决，后续训练需单独冻结方案。']
    atomic(out/'summary.json',report);(out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    atomic(out/'complete.json',dict(status='passed',conditions=984,new_predictions=816,reused_predictions=168,
        summary_sha256=digest(out/'summary.json'),results_sha256=digest(out/'RESULTS.md'),world_model_updates=0))


if __name__=='__main__':main()
