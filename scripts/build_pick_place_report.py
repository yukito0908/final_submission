#!/usr/bin/env python3
"""Build the Japanese Pick and Place submission report from measured artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer,
    Table, TableStyle,
)


ROOT = Path("/workspace")
OUT = ROOT / "output/pdf/pick_place_final_report.pdf"
ART = ROOT / "artifacts/pick_place"
RES = ART / "results"
MEDIA = ART / "media"

pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
FONT = "HeiseiKakuGo-W5"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18 * mm, 10 * mm, "Physical AI - CRANE+ V2 ACT Pick and Place")
    canvas.drawRightString(192 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def image(path: Path, max_width=170 * mm, max_height=100 * mm):
    item = Image(str(path))
    scale = min(max_width / item.imageWidth, max_height / item.imageHeight)
    item.drawWidth = item.imageWidth * scale
    item.drawHeight = item.imageHeight * scale
    return item


def table(rows, widths, header=True, size=8.4):
    style = [
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), size),
        ("LEADING", (0, 0), (-1, -1), size + 4),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94A3B8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEAF7")))
    return Table(rows, colWidths=widths, repeatRows=1 if header else 0, style=TableStyle(style))


def main():
    summary = load(RES / "summary.json")
    evaluation = load(RES / "evaluation_final/evaluation_summary.json")
    video_eval = load(RES / "video_rollout/evaluation_summary.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    doc = BaseDocTemplate(
        str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=18 * mm,
        title="Physical AI - CRANE+ V2 ACT Pick and Place",
        author="Physical AI Final Project",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=footer))

    base = getSampleStyleSheet()
    title = ParagraphStyle("TitleJP", parent=base["Title"], fontName=FONT, fontSize=23,
                           leading=33, alignment=TA_CENTER, textColor=colors.HexColor("#0F172A"))
    subtitle = ParagraphStyle("SubJP", parent=base["Normal"], fontName=FONT, fontSize=11,
                              leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#475569"))
    h1 = ParagraphStyle("H1JP", parent=base["Heading1"], fontName=FONT, fontSize=16,
                        leading=23, spaceBefore=3 * mm, spaceAfter=2.5 * mm,
                        textColor=colors.HexColor("#0F4C81"))
    h2 = ParagraphStyle("H2JP", parent=base["Heading2"], fontName=FONT, fontSize=12,
                        leading=18, spaceBefore=2.5 * mm, spaceAfter=1.5 * mm,
                        textColor=colors.HexColor("#1E3A5F"))
    body = ParagraphStyle("BodyJP", parent=base["BodyText"], fontName=FONT, fontSize=9.1,
                          leading=14.8, alignment=TA_JUSTIFY, spaceAfter=2 * mm,
                          textColor=colors.HexColor("#1F2937"))
    bullet = ParagraphStyle("BulletJP", parent=body, leftIndent=5 * mm, firstLineIndent=-3 * mm)
    caption = ParagraphStyle("CaptionJP", parent=body, fontSize=8, leading=12,
                             alignment=TA_CENTER, textColor=colors.HexColor("#64748B"))

    teacher = summary["teacher"]
    training = summary["training"]
    story = [
        Spacer(1, 22 * mm),
        Paragraph("Physical AI 最終課題発展", title), Spacer(1, 4 * mm),
        Paragraph("CRANE+ V2 Pick and Place", title),
        Paragraph("ACT模倣学習による実把持・持上げ・搬送・配置", title),
        Spacer(1, 8 * mm),
        Paragraph("ROS 2 Humble / Gazebo / MoveIt 2 / LeRobot ACT", subtitle),
        Spacer(1, 7 * mm), image(MEDIA / "video_cam4.png", 165 * mm, 88 * mm),
        Spacer(1, 7 * mm),
        Paragraph("実験日: 2026-10-31　CPU学習・Gazebo物理評価・動画検証済み", subtitle),
        Spacer(1, 5 * mm),
        table([
            ["最終ACT評価", "教師採用", "Dataset", "CPU学習"],
            ["5/5成功", "20 episodes", "12,444 frames", "5,000 steps / 277秒"],
        ], [42.5 * mm] * 4, size=9), PageBreak(),

        Paragraph("要旨", h1),
        Paragraph(
            "Gazebo上のCRANE+ V2が5 cm木製ブロックを把持し、持ち上げ、別位置へ搬送して解放するPick and Placeを実装した。公式MoveItサンプルで動的物体の実把持を確認後、成功教師軌道をrosbag2へ自動収集し、LeRobotDatasetへ変換した。画像を使わず5次元関節状態を入力とするACTをCPUのみで学習し、保存済みモデルをROS 2 trajectory controllerへ接続して評価した。", body),
        Paragraph(
            "正式評価5試行は全工程5/5成功し、平均最大持上げ17.96 cm、平均平面移動32.40 cmだった。成功判定はグリッパ指令だけでなくGazeboのブロック位置で行った。初期VAE版、過度な指令差分制限、時間切れの失敗も保存し、controller追従とaction chunk再計画の相互作用を原因として特定した。", body),
        Paragraph("要件達成状況", h1),
        table([
            ["Phase", "内容", "結果"],
            ["1", "公式サンプルで物理把持・持上げ・搬送を実測", "完了"],
            ["2", "自動教師、rosbag2、物理成功フィルタ", "完了"],
            ["3", "LeRobotDataset、ACT CPU学習", "完了"],
            ["4", "ACTによるGazebo 5試行評価", "5/5成功"],
            ["5", "動画、ソース、README、PDF、数値結果", "完了"],
        ], [18 * mm, 118 * mm, 34 * mm]),
        Paragraph("最終成果物の構成", h2),
        Paragraph("GitHub提出用リポジトリはPick and Placeのコード、データ、学習済みモデル、評価結果、動画、レポートのみで構成した。", body), PageBreak(),

        Paragraph("1. 環境とシステム構成", h1),
        table([
            ["層", "技術", "役割"],
            ["Host", "Windows 11 / Docker Desktop", "講義Docker環境の再利用"],
            ["Simulation", "Gazebo Ignition / CRANE+ V2", "動力学、衝突、物体位置"],
            ["Motion", "ROS 2 Humble / MoveIt 2", "教師Pick and Place"],
            ["Dataset", "rosbag2 / LeRobotDataset", "5次元state/action管理"],
            ["Learning", "ACT / PyTorch / CPU", "32点action chunk予測"],
            ["Evaluation", "ROS 2 / Gazebo pose stream", "工程別の物理成功判定"],
        ], [29 * mm, 57 * mm, 84 * mm]),
        Spacer(1, 4 * mm),
        Paragraph("データフロー", h2),
        table([["公式サンプル", "->", "計測付き教師", "->", "rosbag2", "->", "LeRobotDataset", "->", "ACT", "->", "Gazebo評価"]],
              [22*mm,6*mm,27*mm,6*mm,19*mm,6*mm,31*mm,6*mm,14*mm,6*mm,27*mm], header=False, size=7.6),
        Paragraph("状態・行動仕様", h2),
        Paragraph("順序は joint1, joint2, joint4, joint3, gripper の5次元。工程ラベル、画像、物体位置はACT入力に使用せず、工程ラベルは品質検査と軌道切出しだけに使用した。したがって適用範囲は固定ブロック位置である。", body),
        Paragraph("教師ノード", h2),
        Paragraph("公式MoveIt動作へ reset / approach / grasp / lift / transport / place / retreat / complete の8工程イベント、20 Hz action再配信、購読者準備待ち、Gazebo物体位置計測を追加した。", body),
        image(MEDIA / "video_cam4.png", 165 * mm, 88 * mm),
        Paragraph("図1　赤い開始位置、緑の目標位置、黒い5 cmブロック", caption), PageBreak(),

        Paragraph("2. Phase 1 - 物理把持の実現可能性", h1),
        Paragraph("公式worldのWood cube 5cmは質量0.0565 kg、衝突形状、慣性を持つ動的物体である。公式サンプル完走時のGazebo物体位置を計測した。", body),
        table([
            ["指標", "値"],
            ["初期位置", "(0.000, -0.100, 1.015) m"],
            ["最大高さ", "1.19937 m"],
            ["最大持上げ", "0.18437 m"],
            ["終了位置", "(0.252, 0.004, 1.065) m"],
        ], [60 * mm, 110 * mm]),
        Paragraph("初回はMoveIt/DDS負荷で失敗したが、RViz終了と再起動後に完走した。ブロックが18.44 cm上昇し別位置へ移動したため、グリッパ閉動作だけでなく実把持が成立したと判定した。", body),
        Paragraph("3. Phase 2 - 教師収集", h1),
        table([
            ["項目", "値"],
            ["総試行", str(teacher["attempts"])],
            ["物理成功候補", str(teacher["physically_successful_candidates"])],
            ["実把持失敗", "3"],
            ["工程ラベル不完全", str(teacher["rejected_incomplete_stage_events"])],
            ["最終採用", str(teacher["dataset_selected"])],
            ["物理成功率", f'{teacher["physical_success_rate"] * 100:.2f}%'],
        ], [70 * mm, 100 * mm]),
        Paragraph("3件は教師が終了してもブロック上昇が約4 mm、搬送0 mであり、失敗として除外した。4件は物理成功したがtopic discovery競合で工程イベントが欠けたため除外した。失敗データは削除せずattemptディレクトリへ保存した。", body), PageBreak(),

        Paragraph("4. Datasetと教師品質", h1),
        table([
            ["項目", "値"], ["Repo ID", "local/crane_plus_pick_place"],
            ["Episodes", "20"], ["Frames", "12,444"], ["Frequency", "20 Hz"],
            ["Shape", "state (5,), action (5,)"], ["使用区間", "approach開始からretreat開始直前"],
        ], [52 * mm, 118 * mm]),
        Paragraph(f'採用教師の平均最大持上げは {teacher["lift_m_mean"]*100:.2f} ± {teacher["lift_m_std"]*100:.2f} cm、平均平面移動は {teacher["transport_xy_m_mean"]*100:.2f} ± {teacher["transport_xy_m_std"]*100:.2f} cmだった。24物理成功候補、合計105,277 messageを検査し、8工程が完全な20本だけを学習に使用した。', body),
        image(RES / "teacher_physics.png", 165 * mm, 94 * mm),
        Paragraph("図2　採用教師の物理持上げ量と配置位置", caption),
        Paragraph("5. Phase 3 - ACT CPU学習", h1),
        table([
            ["項目", "最終設定"], ["ACT", "Transformer action chunking, use_vae=false"],
            ["Steps / batch", "5,000 / 8"], ["Learning rate", "1e-4"],
            ["Chunk / action steps", "32 / 32"], ["Model", "dim 128, heads 4, encoder 2, decoder 1, FFN 512"],
            ["Parameters", f'{training["parameter_count"]:,}'], ["Device / threads", "CPU / 12"],
            ["Wall time", f'{training["wall_time_s"]:.2f} s'],
            ["L1 loss", f'initial {training["initial_loss"]:.5f}, final {training["final_loss"]:.5f}, min {training["minimum_loss"]:.5f}'],
        ], [55 * mm, 115 * mm], size=8), PageBreak(),

        Paragraph("5. ACT学習結果と失敗分析", h1),
        image(RES / "training_loss.png", 165 * mm, 92 * mm),
        Paragraph("図3　最終決定論的ACTのCPU学習loss", caption),
        Paragraph("初回VAE版", h2),
        Paragraph("2,000 steps、1,072,579 parameters、179.66秒で学習した。lossは41.109から0.114へ低下したが、開始姿勢で0.05から0.10 radの不要指令があり、2試行のブロック上昇は3.6から3.8 mm、搬送0 mで失敗した。", body),
        Paragraph("過度な指令差分制限", h2),
        Paragraph("決定論的ACTでも、実測関節角との差を1周期0.08 radへ制限するとcontroller追従が遅れた。32 action後の再計画で方策が過去の軌道位置から予測し直し、教師より約3倍遅くなった。40秒試行はブロック上昇4.29 mmで失敗した。", body),
        Paragraph("最終デプロイ条件", h2),
        Paragraph("関節絶対上下限は維持し、差分clipだけを0.50 radへ緩和した。36秒probeは持上げ5.98 cm、搬送8.63 cmで時間切れ、70秒probeは持上げ16.35 cm、搬送33.03 cmで配置まで成功した。これに基づき正式評価を70秒で行った。", body),
        Paragraph("学習lossだけではタスク成功を保証しない。ACT、action queue、再計画周期、低レベルcontrollerを一体で評価する必要がある。", body), PageBreak(),

        Paragraph("6. Phase 4 - ACT正式評価", h1),
        Paragraph("成功条件は、持上げ5 cm以上、平面移動15 cm以上、目標x=0.15から0.35 m、|y|<=0.08 m、グリッパ閉に加え実際のlift、目標領域でのグリッパ再開である。", body),
    ]

    result_rows = [["Trial", "Lift cm", "Move cm", "Final x", "Final y", "Final z", "Result"]]
    for result in evaluation["results"]:
        metrics = result["pose_metrics"]
        xyz = metrics["final_xyz_m"]
        result_rows.append([
            str(result["trial"]), f'{metrics["lift_m"]*100:.2f}',
            f'{metrics["transport_xy_m"]*100:.2f}', f"{xyz[0]:.4f}",
            f"{xyz[1]:.4f}", f"{xyz[2]:.4f}", "成功" if result["success"] else "失敗",
        ])
    story += [
        table(result_rows, [15*mm,24*mm,24*mm,25*mm,25*mm,25*mm,24*mm], size=7.8),
        Spacer(1, 3 * mm),
        Paragraph("正式評価は5/5成功。平均最大持上げ17.96 ± 1.10 cm、平均平面移動32.40 ± 0.53 cm、平均最終位置(0.29698, 0.02953, 1.03000) mだった。", body),
        image(RES / "act_stage_success.png", 150 * mm, 82 * mm),
        Paragraph("図4　ACT工程別成功数（各5/5）", caption),
        Paragraph("配置高さの注意", h2),
        Paragraph("最終zは1.015 mと1.040 mの2群だった。全試行で目標領域へ到達してグリッパは開いたが、3試行は開いた指先付近にブロックが残った。動画試行は最終z=1.015 mでテーブル面へ置かれた。より厳密な静置判定では最終zと静止速度を条件へ加える必要がある。", body), PageBreak(),

        Paragraph("7. 提出動画と物理証拠", h1),
        Paragraph("提出動画は97秒、H.264、1272x668。タイトル5秒、ACT rollout 85秒、結果7秒で構成し、赤い開始パッド、緑の目標パッド、黒いブロック、5工程ラベルを同時表示する。映像は教師ノードではなく保存済みACTモデルの推論である。", body),
        image(MEDIA / "title_preview.png", 165 * mm, 80 * mm),
        Paragraph("図5　動画タイトル - ACT推論であることを明記", caption),
        image(MEDIA / "submission_contact.png", 170 * mm, 82 * mm),
        Paragraph("図6　ACT rolloutの接近・把持・持上げ・搬送・配置", caption),
        Paragraph("録画試行", h2),
        Paragraph(f'持上げ {video_eval["results"][0]["pose_metrics"]["lift_m"]*100:.2f} cm、平面移動 {video_eval["results"][0]["pose_metrics"]["transport_xy_m"]*100:.2f} cm、最終z {video_eval["results"][0]["pose_metrics"]["final_xyz_m"][2]:.3f} mで全工程成功。物体位置CSVと動画が対応している。', body), PageBreak(),

        Paragraph("8. 考察", h1),
        Paragraph("成功要因", h2),
        Paragraph("固定物体位置、再現性の高いMoveIt教師、物理成功データのみの採用により、画像なしの5次元状態でもタスクを学習できた。プロセス終了コードではなく物体位置を収集時・評価時とも測定したことで、グリッパ閉だけを成功とする誤判定を防いだ。", body),
        Paragraph("VAEと決定論ACT", h2),
        Paragraph("多様な人間デモではVAEが有用だが、本データは固定位置の自動教師によるほぼ単峰な軌道である。決定論版はパラメータを約37.5%減らし、推論ばらつきを抑えた。ただしVAE版の長時間学習比較は未実施で、VAE一般を否定する結果ではない。", body),
        Paragraph("一般化範囲", h2),
        Paragraph("画像や物体位置をACT入力に含めないため、ブロック位置変更への一般化は期待できない。次段階ではRGBまたは物体座標を観測へ追加し、位置をランダム化した教師データと未知位置評価が必要である。", body),
        Paragraph("制約と改善", h1),
        Paragraph("- ブロック初期位置は固定。", bullet),
        Paragraph("- 正式評価は5試行で、統計的信頼区間は広い。", bullet),
        Paragraph("- 3/5試行で配置後の高さに2.5 cmの残差がある。", bullet),
        Paragraph("- controller追従のため70秒と教師より遅い。", bullet),
        Paragraph("改善には、解放後退避をDatasetへ含める、最終z・速度判定、chunk全体の時刻付き送信、RGB/物体位置入力、位置ランダム化、30から100試行評価が有効である。", body), PageBreak(),

        Paragraph("9. 再現性と成果物", h1),
        table([
            ["成果物", "場所"],
            ["提出動画", "artifacts/pick_place/media/pick_place_act_submission.mp4"],
            ["ROS 2 source", "ros2_ws/src/crane_plus_pick_place_teacher"],
            ["Dataset", "artifacts/pick_place/data/lerobot"],
            ["最終model", "artifacts/pick_place/models/act_pick_place_deterministic"],
            ["正式評価", "artifacts/pick_place/results/evaluation_final"],
            ["集計・図", "artifacts/pick_place/results"],
        ], [45 * mm, 125 * mm], size=8),
        Paragraph("主要再現コマンド", h2),
        Paragraph("Build: colcon build --packages-select crane_plus_pick_place_teacher --symlink-install", body),
        Paragraph("Training: train_act.py --steps 5000 --batch-size 8 --chunk-size 32 --no-use-vae --threads 12", body),
        Paragraph("Evaluation: evaluate_pick_place_policy.py --trials 5 --run-duration 70 --max-step 0.50", body),
        Paragraph("10. 結論", h1),
        Paragraph("公式サンプルで実把持可能性を確認し、物理成功教師20本からCPU学習したACTが、Gazebo上で把持・持上げ・搬送・配置を5/5回実行した。教師制御をACT成果として扱わず、動画・位置ログ・工程別判定を対応させ、Pick and Place発展課題の実装、学習、評価、提出物作成まで完了した。", body),
    ]

    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
