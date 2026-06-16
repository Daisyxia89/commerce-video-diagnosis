#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

REGRESSION_VIDEO="output/raw_video_regression/clip_8s.mp4"
SOURCE_PRODUCT_ID="3811830677083652287"
SMOKE_WORKSPACE="output/raw_video_smoke_ci"
SMOKE_OUTPUT="$SMOKE_WORKSPACE/combined_smoke_result.json"
SMOKE_DECISION_LOG="$SMOKE_WORKSPACE/smoke_gate_log.json"
WORKFLOW_CONTRACT_CHECKER="user_skills/commerce-video-diagnosis/scripts/check_workflow_contract.py"
TEST_TARGET_RESOLVER="user_skills/commerce-video-diagnosis/scripts/resolve_test_targets.py"
JUNIT_DIR="output/junit"
UNIT_JUNIT_XML="$JUNIT_DIR/junit-unit.xml"
INTEGRATION_JUNIT_XML="$JUNIT_DIR/junit-integration.xml"
SMOKE_JUNIT_XML="$JUNIT_DIR/junit-smoke.xml"
REPLAY_JUNIT_XML="$JUNIT_DIR/junit-replay.xml"
SMOKE_GATE_SCRIPT="user_skills/commerce-video-diagnosis/scripts/run_smoke_gate.py"

mkdir -p "$JUNIT_DIR"

run_contract_stage() {
  echo "Stage: contract"
  python3 "$WORKFLOW_CONTRACT_CHECKER"
}

run_unit_stage() {
  echo "Stage: unit"
  UNIT_TEST_TARGETS="$(python3 "$TEST_TARGET_RESOLVER" --layer unit)"
  pytest -m unit \
    --junitxml "$UNIT_JUNIT_XML" \
    $UNIT_TEST_TARGETS
}

run_integration_stage() {
  echo "Stage: integration"
  INTEGRATION_TEST_TARGETS="$(python3 "$TEST_TARGET_RESOLVER" --layer integration)"
  pytest -m integration \
    --junitxml "$INTEGRATION_JUNIT_XML" \
    $INTEGRATION_TEST_TARGETS
}

run_replay_stage() {
  echo "Stage: replay"
  pytest user_skills/commerce-video-diagnosis/tests/test_replay_pipeline.py \
    --junitxml "$REPLAY_JUNIT_XML"
}

run_smoke_stage() {
  echo "Stage: smoke-gate"
  mkdir -p "$SMOKE_WORKSPACE"
  python3 "$SMOKE_GATE_SCRIPT" \
    --video "$REGRESSION_VIDEO" \
    --video-id raw-video-smoke-8s \
    --source-product-id "$SOURCE_PRODUCT_ID" \
    --workspace "$SMOKE_WORKSPACE" \
    --decision-log "$SMOKE_DECISION_LOG" \
    --include-ocr-regression-summary \
    --output "$SMOKE_OUTPUT"

  SMOKE_MODE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["mode"])' "$SMOKE_DECISION_LOG")"
  SMOKE_REASON_CODE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["reason_code"])' "$SMOKE_DECISION_LOG")"

  echo "Smoke gate mode: $SMOKE_MODE"
  echo "Smoke gate reason_code: $SMOKE_REASON_CODE"

  if [ "$SMOKE_MODE" = "full_smoke_executed" ]; then
    echo "Smoke mode: full smoke + OCR feedback regression"
    SMOKE_TEST_TARGETS="$(python3 "$TEST_TARGET_RESOLVER" --layer smoke)"
    SMOKE_TEST_KEYWORD="$(python3 "$TEST_TARGET_RESOLVER" --layer smoke --format keyword)"
    AIME_SMOKE_WORKSPACE="$SMOKE_WORKSPACE" pytest -m integration \
      --junitxml "$SMOKE_JUNIT_XML" \
      $SMOKE_TEST_TARGETS -k "$SMOKE_TEST_KEYWORD"
    run_replay_stage
  else
    echo "Smoke mode: degraded to pytest-only because provider auth failed"
    echo "Smoke degrade reason_code logged at $SMOKE_DECISION_LOG"
  fi
}

run_all_stages() {
  run_contract_stage
  run_unit_stage
  run_integration_stage
  run_smoke_stage
}

STAGE="${1:-all}"

case "$STAGE" in
  contract)
    run_contract_stage
    ;;
  unit)
    run_unit_stage
    ;;
  integration)
    run_integration_stage
    ;;
  smoke)
    run_smoke_stage
    ;;
  replay)
    run_replay_stage
    ;;
  all)
    run_all_stages
    ;;
  *)
    echo "Usage: $0 [contract|unit|integration|smoke|replay|all]" >&2
    exit 2
    ;;
esac
