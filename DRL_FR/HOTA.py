import os
import sys
import shutil
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# TrackEval 경로 추가
sys.path.append(str(Path(__file__).resolve().parent / "TrackEval"))
from TrackEval.trackeval import eval, datasets, metrics


# === Step 1: 평가 실행 함수 ===
def run_hota_eval(gt_folder, tracker_folder, benchmark='MOT17'): 
    seqmap_path = os.path.join(gt_folder, 'seqmap.txt')
    

    eval_config = {
        'DISPLAY_LESS_PROGRESS': True,
        'OUTPUT_FOLDER': str(Path(gt_folder).parent / "results"),
        'OUTPUT_SUB_FOLDER': '',
        'TRACKERS_TO_EVAL': ['MyTracker'],  
        'TRACKER_SUB_FOLDER': 'data',      
        'METRICS': ['HOTA', 'CLEAR'],
    }

    dataset_config = {
        'GT_FOLDER': gt_folder,
        'TRACKERS_FOLDER': tracker_folder,
        'SEQMAP_FILE': seqmap_path,
        'BENCHMARK': benchmark,
        'SPLIT_TO_EVAL': 'train',
        'SKIP_SPLIT_FOL': True, 
        'TRACKERS_TO_EVAL': ['MyTracker'],
        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt.txt',
        'SKIP_TIMESTAMPS': True
    }
    print("[DEBUG] TRACKERS_FOLDER:", tracker_folder)
    print("[DEBUG] TRACKERS_TO_EVAL:", eval_config['TRACKERS_TO_EVAL'])



    dataset_list = [datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [metrics.HOTA(), metrics.CLEAR()]

    evaluator = eval.Evaluator(eval_config)
    evaluator.evaluate(dataset_list, metrics_list)


# === Step 2: 평가용 데이터 구성 ===
'''
def evaluate_all_sequences(base_data_path, tracker_output_root, save_root):
    gt_dir = Path(save_root) / "gt"
    tracker_dir = Path(save_root) / "trackers"
    tracker_name = "MyTracker"
    tracker_subdir = tracker_dir / tracker_name / "data"
    seqmap_path = gt_dir / "seqmap.txt"

    gt_dir.mkdir(parents=True, exist_ok=True)
    tracker_dir.mkdir(parents=True, exist_ok=True)
    # 트래커 파일 저장 디렉토리 생성 (누락된 부분!)
    tracker_subdir.mkdir(parents=True, exist_ok=True)


    frcnn_folders = [f for f in Path(base_data_path).iterdir() if f.is_dir() and 'FRCNN' in f.name]
    seqmap_lines = []

    for subfolder in frcnn_folders:
        gt_name = subfolder.name

        # === Tracker 결과 불러와 frame 번호 추출
        tracker_src = Path(tracker_output_root) / gt_name / "trackers.txt"
        df_trk = pd.read_csv(tracker_src, header=None, sep=',')  # MOT 포맷: frame, id, x, y, w, h, ...
        tracker_frames = df_trk[0].unique()  # 첫 칼럼이 frame 번호

        # === GT 파일 필터링해서 복사
        gt_src = subfolder / "gt" / "gt.txt"
        df_gt = pd.read_csv(gt_src, header=None, sep=',')  # MOT GT 포맷: frame, id, bb_left, bb_top, w, h, ...
        df_gt_filt = df_gt[df_gt[0].isin(tracker_frames)]

        gt_dest_folder = gt_dir / gt_name
        gt_dest_folder.mkdir(parents=True, exist_ok=True)
        gt_dest = gt_dest_folder / "gt.txt"
        df_gt_filt.to_csv(gt_dest, header=False, index=False)

        # (seqinfo.ini 복사 부분 생략)

        # === Tracker 결과 복사 (원본 톤 그대로)
        tracker_dest = tracker_subdir / f"{gt_name}.txt"
        shutil.copy(tracker_src, tracker_dest)

        # === seqmap.txt에 추가
        seqmap_lines.append(gt_name)


    with open(seqmap_path, 'w') as f:
        for line in seqmap_lines:
            f.write(f"{line}\n")

    print(f"Ground truth, tracker result, and seqmap.txt prepared at: {save_root}")

    run_hota_eval(gt_dir, Path(save_root) / "trackers")
'''
def evaluate_all_sequences(base_data_path, tracker_output_root, save_root, detector='FRCNN'):
    gt_dir = Path(save_root) / "gt"
    tracker_dir = Path(save_root) / "trackers"
    tracker_name = "MyTracker"
    tracker_subdir = tracker_dir / tracker_name / "data"
    seqmap_path = gt_dir / "seqmap.txt"

    # 결과 디렉토리 생성
    gt_dir.mkdir(parents=True, exist_ok=True)
    tracker_subdir.mkdir(parents=True, exist_ok=True)

    # detector 필터링 (필요 없으면 detector=None 으로 호출)
    detector_folders = [
        f for f in Path(base_data_path).iterdir()
        if f.is_dir() and detector in f.name
    ]
    seqmap_lines = []

    for subfolder in detector_folders:
        gt_name = subfolder.name

        # 1) Tracker 결과 복사
        tracker_src = Path(tracker_output_root) / gt_name / "trackers.txt"
        if not tracker_src.exists():
            print(f"[WARN] Tracker not found: {tracker_src}")
            continue
        tracker_dest = tracker_subdir / f"{gt_name}.txt"
        shutil.copy(tracker_src, tracker_dest)

        # tracker에서 사용된 프레임 번호 추출
        df_trk = pd.read_csv(tracker_src, header=None, sep=',')
        tracker_frames = df_trk[0].unique()

        # 2) GT 파일 복사 및 필터링
        gt_src = subfolder / "gt" / "gt.txt"
        if not gt_src.exists():
            print(f"[WARN] GT not found: {gt_src}")
            continue

        gt_dest_folder = gt_dir / gt_name
        gt_dest_folder.mkdir(parents=True, exist_ok=True)

        # --- seqinfo.ini 복사 (TrackEval 필수) ---
        ini_src = subfolder / "seqinfo.ini"
        if ini_src.exists():
            shutil.copy(ini_src, gt_dest_folder / "seqinfo.ini")
        else:
            print(f"[WARN] seqinfo.ini not found for sequence {gt_name}")

        # GT 내용 읽어서 tracker가 등장시킨 프레임만 필터링
        df_gt = pd.read_csv(gt_src, header=None, sep=',')
        df_gt_filt = df_gt[df_gt[0].isin(tracker_frames)]

        gt_dest = gt_dest_folder / "gt.txt"
        df_gt_filt.to_csv(gt_dest, header=False, index=False)

        seqmap_lines.append(gt_name)

    # seqmap.txt 작성
    with open(seqmap_path, 'w') as f:
        for line in seqmap_lines:
            f.write(f"{line}\n")

    print(f"[{detector}] GT and Tracker prepared at: {save_root}")

    # HOTA 평가 실행
    run_hota_eval(gt_dir, tracker_dir)



# === Step 3: 시각화 함수 (HOTA / CLEAR) ===
def visualize_metrics(result_dir, metric="HOTA"):
    result_dir = Path(result_dir)
    all_results = {}

    for seq_folder in result_dir.iterdir():
        summary_file = seq_folder / f"{metric}_summary.json"
        if not summary_file.exists():
            print(f" {summary_file} 없음, 건너뜀")
            continue
        with open(summary_file) as f:
            data = json.load(f)
            all_results[seq_folder.name] = data.get(metric, 0)

    # 시각화 - 만약 데이터가 없으면 중단
    if not all_results:
        print("No results to visualize.")
        return

    # 모든 역시에 대해 bar chart
    seqs = list(all_results.keys())
    scores = [all_results[seq] for seq in seqs]

    plt.figure(figsize=(12, 5))
    plt.bar(seqs, scores, color='skyblue')
    plt.ylabel(f"{metric} Score")
    plt.xlabel("Sequence")
    plt.title(f"{metric} Score per Sequence")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === 전체 평균 ===
    avg_score = sum(scores) / len(scores)
    print(f"\n[Mean {metric}] {avg_score:.4f}")


# === Step 4: 메인 실행 ===
if __name__ == "__main__":
    evaluate_all_sequences(
        base_data_path='/home/hyhy/Datasets/FR_Dataset/MOT17/train',
        tracker_output_root='/home/hyhy/Desktop/SYD_DtoS/DRL_FR/yolov7_object_tracking/runs/MOT_ds',
        save_root='/home/hyhy/Desktop/SYD_DtoS/DRL_FR/result', detector='FRCNN'
    )

    visualize_metrics("/home/hyhy/Desktop/SYD_DtoS/DRL_FR/result", metric="HOTA")
