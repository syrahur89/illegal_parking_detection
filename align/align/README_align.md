# README_align

## 입력 파일 (반드시 확인)
- 템플릿 랜드마크: C:/Users/yousu/OneDrive/바탕 화면/250910_랜드마크 매핑 csv/template_points_0919수정.csv
  - CSV with columns including: LM (id/name), x, y
- 랜드마크 폴더: C:/Users/yousu/OneDrive/바탕 화면/250910_랜드마크 매핑 csv/4점 모음집
  - 각 이미지별 `<image_id>.csv` 파일 (columns include LM,x,y)
- 마스크 폴더: C:/Users/yousu/OneDrive/바탕 화면/250910_랜드마크 매핑 csv/dataset_master_per_image
  - GeoJSON files (e.g. zoneA_no_parking.geojson, legal_areas.geojson)

## 출력(align/ 루트)
- align/H/<image_id>.npy  (3x3 float64)
- align/quality/<image_id>.json
- align/polygons/<image_id>_<source_mask>.geojson (좌표계: 템플릿 픽셀)
- align/previews/<image_id>_overlay.jpg
- logs/runtime.log
- logs/failures.txt
- logs/skipped.txt
- summary/metrics.csv
- summary/metrics_summary.json
- summary/thresholds_used.json

## 실행 예시
$ python your_script_name.py
(또는 필요시 경로를 스크립트 상단에서 수정)

## 사용 임계값 (현재 파일 thresholds_used.json 참고)
- RMSE_MAX: 15.0
- MIN_INLIERS: 4
- Z_SCORE_THRESHOLD: 4.0
- RANSAC_REPROJ_THRESH: 5.0

## 좌표계
- 입력 랜드마크: 이미지 좌표 (각 이미지의 CSV에 저장된 좌표)
- 템플릿 랜드마크: 템플릿 픽셀 좌표 (templates/...)
- 출력 폴리곤: 템플릿 픽셀 좌표계로 정합되어 저장

