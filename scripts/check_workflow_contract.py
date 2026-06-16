from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / '.github/workflows/commerce-video-diagnosis-regression.yml'
TEMPLATE = REPO_ROOT / 'ci/commerce-video-diagnosis-regression.template.yaml'
LOCAL_CI = REPO_ROOT / 'user_skills/commerce-video-diagnosis/scripts/local_ci.sh'
TEST_TARGET_RESOLVER = REPO_ROOT / 'user_skills/commerce-video-diagnosis/scripts/resolve_test_targets.py'
TEST_TARGET_MANIFEST = REPO_ROOT / 'user_skills/commerce-video-diagnosis/references/test_targets.json'
CLI_CONTRACT_MANIFEST = REPO_ROOT / 'user_skills/commerce-video-diagnosis/references/script_cli_contracts.json'


def _read(path: Path) -> str:
    if not path.is_file():
        raise AssertionError(f'missing file: {path}')
    return path.read_text(encoding='utf-8')


def _assert_order(text: str, parts: list[str], label: str) -> None:
    pos = -1
    for part in parts:
        idx = text.find(part)
        if idx == -1:
            raise AssertionError(f'{label}: missing snippet: {part}')
        if idx < pos:
            raise AssertionError(f'{label}: order violation around: {part}')
        pos = idx


def _assert_github_workflow_contract(text: str) -> None:
    resolver_snippet = 'resolve_test_targets.py --layer unit'
    if resolver_snippet not in text:
        raise AssertionError(f'github-workflow: missing SSOT resolver snippet {resolver_snippet}')
    integration_resolver_snippet = 'resolve_test_targets.py --layer integration'
    if integration_resolver_snippet not in text:
        raise AssertionError(f'github-workflow: missing SSOT resolver snippet {integration_resolver_snippet}')
    _assert_order(
        text,
        [
            'Stage unit / run regression via pytest markers',
            'Publish unit junit report',
            'Upload pytest junit xml',
        ],
        'github-workflow/unit',
    )
    _assert_order(
        text,
        [
            '--junitxml "$AIME_JUNIT_DIR/junit-unit.xml"',
            'name: junit / unit',
            'name: pytest-junit-xml',
        ],
        'github-workflow/unit-anchors',
    )
    _assert_order(
        text,
        [
            'Stage integration / run regression via pytest markers',
            'Publish integration junit report',
            'Upload pytest junit xml',
        ],
        'github-workflow/integration',
    )
    _assert_order(
        text,
        [
            '--junitxml "$AIME_JUNIT_DIR/junit-integration.xml"',
            'name: junit / integration',
            'name: pytest-junit-xml',
        ],
        'github-workflow/integration-anchors',
    )
    smoke_resolver_snippet = 'resolve_test_targets.py --layer smoke'
    if smoke_resolver_snippet not in text:
        raise AssertionError(f'github-workflow: missing smoke SSOT resolver snippet {smoke_resolver_snippet}')
    _assert_order(
        text,
        [
            'Stage smoke / run smoke gate with auth fallback',
            'Stage smoke / resolve smoke gate decision',
            'Stage smoke / emit smoke gate summary',
            'Stage smoke / assert unified smoke output exists',
            'Stage smoke / assert full smoke outputs via integration marker',
            'SMOKE_TEST_TARGETS="$(python3 user_skills/commerce-video-diagnosis/scripts/resolve_test_targets.py --layer smoke)"',
            'SMOKE_TEST_KEYWORD="$(python3 user_skills/commerce-video-diagnosis/scripts/resolve_test_targets.py --layer smoke --format keyword)"',
            'Publish smoke junit report',
            'Upload smoke artifacts',
        ],
        'github-workflow/smoke',
    )
    _assert_order(
        text,
        [
            'run_smoke_gate.py',
            '--junitxml "$AIME_JUNIT_DIR/junit-smoke.xml"',
            'name: junit / smoke',
            'name: raw-video-smoke-ci-artifacts',
        ],
        'github-workflow/smoke-anchors',
    )
    if 'smoke_gate_log.json' not in text:
        raise AssertionError('github-workflow: missing smoke gate log artifact')


def _assert_ci_template_contract(text: str) -> None:
    resolver_snippet = 'resolve_test_targets.py --layer unit'
    if resolver_snippet not in text:
        raise AssertionError(f'ci-template: missing SSOT resolver snippet {resolver_snippet}')
    integration_resolver_snippet = 'resolve_test_targets.py --layer integration'
    if integration_resolver_snippet not in text:
        raise AssertionError(f'ci-template: missing SSOT resolver snippet {integration_resolver_snippet}')
    _assert_order(
        text,
        [
            'stage unit / run regression via pytest markers',
            'publish unit junit',
        ],
        'ci-template/unit',
    )
    _assert_order(
        text,
        [
            '--junitxml ${AIME_JUNIT_DIR}/junit-unit.xml',
            '${AIME_JUNIT_DIR}/junit-unit.xml',
        ],
        'ci-template/unit-anchors',
    )
    _assert_order(
        text,
        [
            'stage integration / run regression via pytest markers',
            'publish integration junit',
        ],
        'ci-template/integration',
    )
    _assert_order(
        text,
        [
            '--junitxml ${AIME_JUNIT_DIR}/junit-integration.xml',
            '${AIME_JUNIT_DIR}/junit-integration.xml',
        ],
        'ci-template/integration-anchors',
    )
    smoke_resolver_snippet = 'resolve_test_targets.py --layer smoke'
    if smoke_resolver_snippet not in text:
        raise AssertionError(f'ci-template: missing smoke SSOT resolver snippet {smoke_resolver_snippet}')
    _assert_order(
        text,
        [
            'stage smoke / run smoke gate with auth fallback',
            'stage smoke / resolve smoke gate decision',
            'stage smoke / assert unified smoke output exists',
            'stage smoke / assert full smoke outputs via integration marker',
            'resolve_test_targets.py --layer smoke',
            'resolve_test_targets.py --layer smoke --format keyword',
            'publish smoke junit',
        ],
        'ci-template/smoke',
    )
    _assert_order(
        text,
        [
            'run_smoke_gate.py',
            '--junitxml ${AIME_JUNIT_DIR}/junit-smoke.xml',
            '${AIME_JUNIT_DIR}/junit-smoke.xml',
        ],
        'ci-template/smoke-anchors',
    )
    if 'smoke_gate_log.json' not in text:
        raise AssertionError('ci-template: missing smoke gate log artifact')


def _assert_local_ci_contract(text: str) -> None:
    resolver_snippet = 'TEST_TARGET_RESOLVER="user_skills/commerce-video-diagnosis/scripts/resolve_test_targets.py"'
    if resolver_snippet not in text:
        raise AssertionError(f'local-ci: missing SSOT resolver snippet {resolver_snippet}')
    for required_stage in ['run_contract_stage()', 'run_unit_stage()', 'run_integration_stage()', 'run_smoke_stage()', 'run_replay_stage()']:
        if required_stage not in text:
            raise AssertionError(f'local-ci: missing stage function {required_stage}')
    if 'Usage: $0 [contract|unit|integration|smoke|replay|all]' not in text:
        raise AssertionError('local-ci: missing staged usage output')
    _assert_order(
        text,
        [
            'run_contract_stage() {',
            'python3 "$WORKFLOW_CONTRACT_CHECKER"',
            'run_unit_stage() {',
            'pytest -m unit \\',
            '--junitxml "$UNIT_JUNIT_XML"',
            'run_integration_stage() {',
            'pytest -m integration \\',
            '--junitxml "$INTEGRATION_JUNIT_XML"',
            'run_smoke_stage() {',
        ],
        'local-ci/pytest-layers',
    )
    smoke_resolver_snippet = 'python3 "$TEST_TARGET_RESOLVER" --layer smoke'
    if smoke_resolver_snippet not in text:
        raise AssertionError(f'local-ci: missing smoke SSOT resolver snippet {smoke_resolver_snippet}')
    _assert_order(
        text,
        [
            'python3 "$SMOKE_GATE_SCRIPT" \\',
            '--decision-log "$SMOKE_DECISION_LOG"',
            'SMOKE_MODE="$(python3 -c',
            'echo "Smoke gate mode: $SMOKE_MODE"',
            'echo "Smoke gate reason_code: $SMOKE_REASON_CODE"',
            'echo "Smoke mode: full smoke + OCR feedback regression"',
            'SMOKE_TEST_TARGETS="$(python3 "$TEST_TARGET_RESOLVER" --layer smoke)"',
            'SMOKE_TEST_KEYWORD="$(python3 "$TEST_TARGET_RESOLVER" --layer smoke --format keyword)"',
            'AIME_SMOKE_WORKSPACE="$SMOKE_WORKSPACE" pytest -m integration \\',
            '--junitxml "$SMOKE_JUNIT_XML"',
            '$SMOKE_TEST_TARGETS -k "$SMOKE_TEST_KEYWORD"',
            'echo "Smoke mode: degraded to pytest-only because provider auth failed"',
            'run_all_stages() {',
            '  run_contract_stage',
            '  run_unit_stage',
            '  run_integration_stage',
            '  run_smoke_stage',
            'case "$STAGE" in',
        ],
        'local-ci/smoke',
    )
    if 'smoke_gate_log.json' not in text:
        raise AssertionError('local-ci: missing smoke gate log output')


def _script_uses_argparse(path: Path) -> bool:
    text = _read(path)
    argparse_markers = [
        'import argparse',
        'from argparse import',
        'argparse.ArgumentParser(',
    ]
    return any(marker in text for marker in argparse_markers)


def _literal_bool(node: ast.AST | None) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _literal_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def scan_required_cli_args(script_path: Path) -> list[str]:
    source = _read(script_path)
    tree = ast.parse(source, filename=str(script_path))

    parser_names: set[str] = set()
    required_args: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id == 'argparse' and func.attr == 'ArgumentParser':
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            parser_names.add(target.id)
            elif isinstance(func, ast.Name) and func.id == 'ArgumentParser':
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        parser_names.add(target.id)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == 'add_argument'):
            continue
        if not isinstance(func.value, ast.Name) or func.value.id not in parser_names:
            continue

        required_kw = None
        for kw in node.keywords:
            if kw.arg == 'required':
                required_kw = _literal_bool(kw.value)
                break
        if required_kw is not True:
            continue

        option_strings: list[str] = []
        for arg in node.args:
            value = _literal_str(arg)
            if value is not None and value.startswith('--'):
                option_strings.append(value)
        if not option_strings:
            continue
        required_args.append(option_strings[0])

    return required_args


def _assert_cli_contract_manifest(text: str) -> None:
    payload = json.loads(text)
    scripts = payload.get('argparse_required_scripts')
    if not isinstance(scripts, list) or not scripts:
        raise AssertionError('cli-contract-manifest: argparse_required_scripts must be a non-empty list')

    required_paths = [
        'user_skills/commerce-video-diagnosis/scripts/run_extractor.py',
        'user_skills/commerce-video-diagnosis/scripts/run_two_stage_smoke.py',
        'user_skills/commerce-video-diagnosis/scripts/run_raw_video_smoke.py',
        'user_skills/commerce-video-diagnosis/scripts/run_smoke_gate.py',
        'user_skills/commerce-video-diagnosis/scripts/run_raw_video_ocr_feedback_regression.py',
        'user_skills/commerce-video-diagnosis/scripts/resolve_test_targets.py',
        'user_skills/commerce-video-diagnosis/scripts/assert_full_smoke_outputs.py',
        'user_skills/commerce-video-diagnosis/scripts/build_request.py',
        'user_skills/commerce-video-diagnosis/scripts/check_workflow_contract.py',
    ]
    seen_paths: list[str] = []
    for entry in scripts:
        if not isinstance(entry, dict):
            raise AssertionError('cli-contract-manifest: each entry must be an object')
        path = entry.get('path')
        if not isinstance(path, str) or not path:
            raise AssertionError('cli-contract-manifest: each entry must declare non-empty path')
        seen_paths.append(path)
        probe_args = entry.get('probe_args')
        if not isinstance(probe_args, list):
            raise AssertionError(f'cli-contract-manifest: {path} missing probe_args list')
        if entry.get('must_fail_on_unknown_args') is not True:
            raise AssertionError(f'cli-contract-manifest: {path} must set must_fail_on_unknown_args=true')

        script_path = REPO_ROOT / path
        if not script_path.is_file():
            raise AssertionError(f'cli-contract-manifest: declared script does not exist: {path}')
        if not _script_uses_argparse(script_path):
            raise AssertionError(f'cli-contract-manifest: declared script has no argparse CLI: {path}')

        statically_required_args = scan_required_cli_args(script_path)
        for required_arg in statically_required_args:
            if required_arg not in probe_args:
                raise AssertionError(
                    f'cli-contract-manifest: {path} probe_args missing statically required arg {required_arg}'
                )

    for required_path in required_paths:
        if required_path not in seen_paths:
            raise AssertionError(f'cli-contract-manifest: missing required script {required_path}')

    scripts_dir = REPO_ROOT / 'user_skills/commerce-video-diagnosis/scripts'
    cli_script_paths = []
    for script_path in sorted(scripts_dir.glob('*.py')):
        if _script_uses_argparse(script_path):
            rel = script_path.relative_to(REPO_ROOT).as_posix()
            cli_script_paths.append(rel)

    missing_from_manifest = [path for path in cli_script_paths if path not in seen_paths]
    if missing_from_manifest:
        raise AssertionError(
            'cli-contract-manifest: argparse CLI scripts missing from manifest: '
            + ', '.join(missing_from_manifest)
        )


def validate_contracts() -> None:
    workflow = _read(WORKFLOW)
    template = _read(TEMPLATE)
    local_ci = _read(LOCAL_CI)
    _read(TEST_TARGET_RESOLVER)
    manifest = _read(TEST_TARGET_MANIFEST)
    cli_manifest = _read(CLI_CONTRACT_MANIFEST)

    for required_manifest_key in ['"unit"', '"integration"', '"smoke"']:
        if required_manifest_key not in manifest:
            raise AssertionError(f'test-target-manifest: missing key {required_manifest_key}')

    _assert_cli_contract_manifest(cli_manifest)
    _assert_github_workflow_contract(workflow)
    _assert_ci_template_contract(template)
    _assert_local_ci_contract(local_ci)

    required_markers = [
        'summary_marker: SMOKE_FULL_MODE',
        'summary_marker: SMOKE_DEGRADED_AUTH_FAILURE',
        'summary_marker: WORKFLOW_FULL_SMOKE_EXECUTED',
        'summary_marker: WORKFLOW_SMOKE_DEGRADED_AUTH_FAILURE',
    ]
    for marker in required_markers:
        if marker not in workflow:
            raise AssertionError(f'github-workflow: missing summary marker {marker}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate workflow/local-ci/template contract for commerce-video-diagnosis.')
    return parser.parse_args()


def main() -> int:
    parse_args()
    validate_contracts()
    print('workflow contract check passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
