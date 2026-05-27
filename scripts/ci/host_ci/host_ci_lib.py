#! /usr/bin/env python3
#
# (C) Copyright 2011- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
"""Shared host-side CI orchestration for OpenIFS.

This module contains the common control/test/bit-compare workflow used by
both the generic host CI entrypoint and the ECMWF-HPC variant. The concrete
differences are expressed as small execution profiles instead of duplicated
driver scripts.
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

_SHARED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared")
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)

import ci_lib
import find_py_packages
import read_yml_config
import setup_logging
import shared_helpers


BITCOMPARE_SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "docker_ci", "openifs_branch_bitcompare.py")
)

STATUS_PASS = "PASS - Complete and Successful"
STATUS_BUILD_FAILED = "FAILED - Build did not complete"
STATUS_TEST_FAILED = "FAILED - Test did not complete"
STATUS_NO_NORMS = "FAILED - No NORMS produced"
STATUS_NOT_RUN = "SKIPPED - Not run"
STATUS_REUSED_NORMS = "SKIPPED - Reused cached NORMS"

VALID_COMPILERS = {"gnu", "intel"}


def _fresh_stage_statuses():
    return {"build": STATUS_NOT_RUN, "test": STATUS_NOT_RUN}


def _stages_passed(stage_statuses):
    return (
        stage_statuses["build"] == STATUS_PASS and
        stage_statuses["test"] == STATUS_PASS
    )


def _host_validate(config):
    compiler_version = str(config.get("compiler_version", "")).strip()
    if not compiler_version:
        raise ValueError("Invalid config value for 'compiler_version'. Expected a non-empty value")


def _host_cache_key(config):
    return f"gcc{config['compiler_version']}"


def _host_log_suffix(config):
    return f"gcc{config['compiler_version']}"


def _host_summary_label(_config):
    return "host"


def _host_prepare_source(staged_src, _config):
    shared_helpers.patch_oifs_home(staged_src)


def _detect_runtime_environment():
    has_slurm = all(shutil.which(cmd) for cmd in ("srun", "salloc"))
    has_modules = bool(
        os.environ.get("LMOD_CMD") or
        os.environ.get("MODULESHOME") or
        shutil.which("modulecmd")
    )
    if has_slurm and has_modules:
        return "ecmwf_hpc"
    if shutil.which("docker"):
        return "docker_host"
    return "standard_host"


def _enforce_runtime_environment(config, profile_name):
    if config.get("allow_unsupported_environment", False):
        return

    detected = _detect_runtime_environment()
    if profile_name == "ecmwf_hpc" and detected != "ecmwf_hpc":
        raise EnvironmentError(
            "The ECMWF-HPC CI profile can only run on an ECMWF-HPC system. "
            f"Detected environment: {detected}. "
            "Set allow_unsupported_environment: True only if you intentionally "
            "want to bypass this guard."
        )


def _compiler_value(config):
    compiler = str(config.get("compiler", "")).strip().lower()
    if compiler not in VALID_COMPILERS:
        raise ValueError(
            "Invalid config value for 'compiler'. Expected one of: gnu, intel"
        )
    return compiler


def _hpc_validate(config):
    _compiler_value(config)


def _hpc_cache_key(config):
    return _compiler_value(config)


def _hpc_arch_path(config, staged_src=None):
    arch_path = str(config.get("arch_path", "")).strip()
    if arch_path:
        return arch_path

    default_path = f"./arch/ecmwf/hpc2020/{_compiler_value(config)}/default"
    if staged_src is not None:
        default_abspath = os.path.join(staged_src, default_path.removeprefix("./"))
        if os.path.exists(default_abspath):
            return default_path

    if _compiler_value(config) == "gnu":
        return default_path

    raise ValueError(
        "No default arch_path exists for compiler 'intel'. Set 'arch_path' to a concrete "
        "leaf such as ./arch/ecmwf/hpc2020/intel/2021.4.0/intel-mpi/2021.4.0"
    )


def _hpc_prepare_source(staged_src, config):
    config_file = shared_helpers.patch_oifs_home(staged_src)
    arch_path = _hpc_arch_path(config, staged_src)
    arch_abspath = os.path.join(staged_src, arch_path.removeprefix("./"))
    if not os.path.exists(arch_abspath):
        raise FileNotFoundError(
            f"Configured arch_path '{arch_path}' does not exist under staged source {staged_src}"
        )

    with open(config_file, encoding="utf-8") as f:
        content = f.read()

    replacements = {
        r'^export OIFS_HOST=.*$': 'export OIFS_HOST="ecmwf"',
        r'^export OIFS_PLATFORM=.*$': 'export OIFS_PLATFORM="hpc2020"',
        r'^export OIFS_ARCH=.*$': f'export OIFS_ARCH="{arch_path}"',
    }
    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    with open(config_file, "w", encoding="utf-8") as f:
        f.write(content)


def _hpc_build_test_commands(config, source_cmd, build_output_path, test_output_path):
    extra_flags = config.get("openifs_test_extra_flags", "").strip()
    test_env_prefix = ci_lib.TEST_ENV_PREFIX
    cb_cmd = (
        f"set -o pipefail; {source_cmd} && {test_env_prefix} "
        f"$OIFS_TEST/openifs-test.sh -cb {extra_flags} "
        f"2>&1 | tee {build_output_path}"
    )
    t_cmd = (
        f"set -o pipefail; {source_cmd} && {test_env_prefix} "
        f"$OIFS_TEST/openifs-test.sh -t 2>&1 | tee {test_output_path}"
    )
    return cb_cmd, t_cmd


def _hpc_log_lines(config):
    return [
        f"Compiler family : {_compiler_value(config)}",
        f"Arch override   : {_hpc_arch_path(config)}",
    ]


def _hpc_log_suffix(config):
    return _compiler_value(config)


def _hpc_summary_label(_config):
    return "ecmwf-hpc"


PROFILES = {
    "host": {
        "name": "host",
        "banner": "Host OpenIFS CI",
        "description": (
            "Host CI test for OpenIFS: stage control + test branches, run "
            "openifs-test.sh -cbt in each, and bit-compare SAVED_NORMS "
            "directly on the host."
        ),
        "config_help": "YAML configuration file (see config/ci_test_host.yml)",
        "validate": _host_validate,
        "cache_key": _host_cache_key,
        "build_commands": ci_lib.build_test_commands,
        "prepare_source": _host_prepare_source,
        "log_suffix": _host_log_suffix,
        "summary_label": _host_summary_label,
    },
    "ecmwf_hpc": {
        "name": "ecmwf_hpc",
        "banner": "ECMWF-HPC OpenIFS CI",
        "description": (
            "ECMWF-HPC CI test for OpenIFS: stage control + test branches, "
            "run openifs-test.sh with the HPC2020 arch for gnu or intel, "
            "and bit-compare SAVED_NORMS directly on the host."
        ),
        "config_help": "YAML configuration file (see config/ci_test_ecmwf_hpc.yml)",
        "validate": _hpc_validate,
        "cache_key": _hpc_cache_key,
        "build_commands": _hpc_build_test_commands,
        "prepare_source": _hpc_prepare_source,
        "log_lines": _hpc_log_lines,
        "log_suffix": _hpc_log_suffix,
        "summary_label": _hpc_summary_label,
    },
}


def _profile(profile_name):
    try:
        return PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(f"Unknown host CI profile '{profile_name}'") from exc


def _validate_profile_config(config, profile_name):
    declared_profile = str(config.get("execution_profile", "")).strip()
    if declared_profile and declared_profile != profile_name:
        raise ValueError(
            f"Config execution_profile '{declared_profile}' does not match requested "
            f"profile '{profile_name}'"
        )
    _profile(profile_name)["validate"](config)


def parse_arguments(profile_name):
    profile = _profile(profile_name)
    parser = argparse.ArgumentParser(
        description=profile["description"],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=str, required=True,
                        help=profile["config_help"])
    return parser.parse_args()


def _export_lasttest_log(staged_src, ci_reports, label):
    logger = logging.getLogger(__name__)
    src = os.path.join(staged_src, "build", "Testing", "Temporary", "LastTest.log")
    dst = os.path.join(ci_reports, f"openifs_lasttest_output_{label}.txt")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        logger.info(f"Captured {label} LastTest.log -> {dst}")
    else:
        logger.warning(f"No LastTest.log captured for {label} (file missing)")


def run_openifs_tests(staged_src, config, ci_reports, label, profile_name):
    logger = logging.getLogger(__name__)
    profile = _profile(profile_name)
    stage_statuses = _fresh_stage_statuses()

    profile.get("prepare_source", _host_prepare_source)(staged_src, config)

    os.makedirs(ci_reports, exist_ok=True)
    build_output = os.path.join(ci_reports, f"openifs_build_output_{label}.txt")
    test_output = os.path.join(ci_reports, f"openifs_test_output_{label}.txt")

    source_cmd = f"source {staged_src}/oifs-config.edit_me.sh"
    cb_cmd, t_cmd = profile["build_commands"](config, source_cmd, build_output, test_output)

    try:
        logger.info(f"Configure + build for {label} in {staged_src}")
        for line in profile.get("log_lines", lambda _: [])(config):
            logger.info(line)
        try:
            subprocess.run(["bash", "-lc", cb_cmd], cwd=staged_src, check=True)
            stage_statuses["build"] = STATUS_PASS
        except subprocess.CalledProcessError:
            stage_statuses["build"] = STATUS_BUILD_FAILED
            stage_statuses["test"] = "SKIPPED - Build failed"
            return stage_statuses

        logger.info(f"Running ctest for {label} in {staged_src}")
        try:
            subprocess.run(["bash", "-lc", t_cmd], cwd=staged_src, check=True)
            stage_statuses["test"] = STATUS_PASS
        except subprocess.CalledProcessError:
            stage_statuses["test"] = STATUS_TEST_FAILED
    finally:
        _export_lasttest_log(staged_src, ci_reports, label)

    return stage_statuses


def find_saved_norms_root(staged_src):
    logger = logging.getLogger(__name__)
    for root, _dirs, files in os.walk(staged_src):
        if "SAVED_NORMS" in files:
            test_root = os.path.dirname(root)
            logger.info(f"SAVED_NORMS root at {test_root}")
            return test_root
    raise FileNotFoundError(
        f"No SAVED_NORMS found under {staged_src} - tests did not produce reference NORMS"
    )


def export_control_norms(test_root, control_tarball):
    logger = logging.getLogger(__name__)
    os.makedirs(os.path.dirname(control_tarball), exist_ok=True)
    logger.info(f"Bundling control SAVED_NORMS -> {control_tarball}")
    with tarfile.open(control_tarball, "w:gz") as tar:
        for name in sorted(os.listdir(test_root)):
            tar.add(os.path.join(test_root, name), arcname=name)


def run_control_phase(config, build_dir, control_tarball, ci_reports, profile_name):
    logger = logging.getLogger(__name__)

    if config.get("reuse_control_if_present", False) and os.path.exists(control_tarball):
        logger.info("=" * 70)
        logger.info(f"Reusing existing control tarball: {control_tarball}")
        logger.info("(reuse_control_if_present=True; delete the tarball or set the flag")
        logger.info(" to False to force a fresh control run)")
        logger.info("=" * 70)
        return "reused", {"build": STATUS_REUSED_NORMS, "test": STATUS_REUSED_NORMS}

    stage_statuses = _fresh_stage_statuses()
    try:
        clone_dir, _ = shared_helpers.stage_branch_source(
            config, "control", build_dir, __file__,
        )
        stage_statuses = run_openifs_tests(clone_dir, config, ci_reports, "control", profile_name)
        if not _stages_passed(stage_statuses):
            return "failed", stage_statuses
        test_root = find_saved_norms_root(clone_dir)
        export_control_norms(test_root, control_tarball)
        return "ok", stage_statuses
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logger.error(f"Control phase FAILED: {e}")
        if stage_statuses["build"] == STATUS_NOT_RUN:
            stage_statuses["build"] = STATUS_BUILD_FAILED
        if _stages_passed(stage_statuses):
            stage_statuses["test"] = STATUS_NO_NORMS
        return "failed", stage_statuses


def run_test_phase(config, build_dir, ci_reports, profile_name):
    clone_dir, _ = shared_helpers.stage_branch_source(
        config, "test", build_dir, __file__,
    )
    stage_statuses = run_openifs_tests(clone_dir, config, ci_reports, "test", profile_name)
    if not _stages_passed(stage_statuses):
        return None, stage_statuses
    try:
        test_root = find_saved_norms_root(clone_dir)
    except FileNotFoundError:
        stage_statuses["test"] = STATUS_NO_NORMS
        return None, stage_statuses
    return test_root, stage_statuses


def compare_norms(test_root, control_tarball, report_path, build_dir):
    logger = logging.getLogger(__name__)

    control_extract_dir = os.path.join(build_dir, "control_saved_norms_extracted")
    if os.path.exists(control_extract_dir):
        shutil.rmtree(control_extract_dir)
    os.makedirs(control_extract_dir, exist_ok=True)
    logger.info(f"Extracting {control_tarball} -> {control_extract_dir}")
    with tarfile.open(control_tarball, "r:gz") as tar:
        tar.extractall(control_extract_dir)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    logger.info(f"Running {BITCOMPARE_SCRIPT}")
    result = subprocess.run(
        ["python3", BITCOMPARE_SCRIPT,
         control_extract_dir, test_root,
         "--report", report_path],
    )
    return result.returncode == 0


def run_profile(profile_name):
    script_start = time.time()
    timings = {}
    profile = _profile(profile_name)

    cli_args = parse_arguments(profile_name)

    find_py_packages.main(["yaml"])

    config = read_yml_config.main(cli_args.config)
    _validate_profile_config(config, profile_name)
    _enforce_runtime_environment(config, profile_name)

    build_dir = os.path.expanduser(os.path.expandvars(config["openifs_build_host_dir"]))
    os.makedirs(build_dir, exist_ok=True)

    log_dir = os.path.join(build_dir, "host_ci_logfiles")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"log_ci_{config['openifs_version']}_{profile['log_suffix'](config)}.log",
    )
    setup_logging.main(log_path)
    logger = logging.getLogger(__name__)

    ci_reports = os.path.expanduser(os.path.expandvars(config["ci_reports"]))
    control_dir = os.path.expanduser(os.path.expandvars(config["control_saved_norms_dir"]))
    control_tarball = os.path.join(
        control_dir,
        ci_lib.control_tarball_name(config, profile["cache_key"](config)),
    )
    report_path = os.path.join(ci_reports, ci_lib.report_filename(config, __file__))

    logger.info("=" * 70)
    logger.info(profile["banner"])
    logger.info("=" * 70)
    for line in profile.get("log_lines", lambda _: [])(config):
        logger.info(line)

    with shared_helpers.timer(f"Control phase ({config['control_branch']})", timings, "control-branch"):
        control_status, control_stage_statuses = run_control_phase(
            config, build_dir, control_tarball, ci_reports, profile_name,
        )

    test_branch = config["test_branch"]
    test_label = test_branch or "auto-resolved local source"
    test_root = None
    test_stage_statuses = _fresh_stage_statuses()
    test_phase_failed = False
    try:
        with shared_helpers.timer(f"Test phase ({test_label})", timings, "test-branch"):
            test_root, test_stage_statuses = run_test_phase(config, build_dir, ci_reports, profile_name)
            test_phase_failed = not _stages_passed(test_stage_statuses)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logger.error(f"Test phase FAILED: {e}")
        test_phase_failed = True
        test_stage_statuses["build"] = STATUS_BUILD_FAILED

    if test_phase_failed:
        ci_lib.write_synthetic_report(
            report_path,
            f"Execution profile: {profile['summary_label'](config)}\n"
            f"CI banner: {profile['banner']}\n\n"
            "Test phase FAILED during configure+build or ctest. "
            "See the uploaded BUILD OUTPUT and CTEST OUTPUT artifacts for the cause.",
        )
        bit_compare_status = "SKIPPED"
        bit_compare_skip_reason = "No test NORMS"
        timings["norms_compare"] = 0
    elif control_status == "failed":
        logger.warning("Skipping bit-comparison — control phase did not produce SAVED_NORMS")
        ci_lib.write_synthetic_report(
            report_path,
            f"Execution profile: {profile['summary_label'](config)}\n"
            f"CI banner: {profile['banner']}\n\n"
            "Control phase FAILED — no SAVED_NORMS to compare against. "
            "Test phase ran to completion; see uploaded artifacts for details.",
        )
        bit_compare_status = "SKIPPED"
        bit_compare_skip_reason = "No control NORMS"
        timings["norms_compare"] = 0
    else:
        with shared_helpers.timer(
            f"NORMS comparison ({profile['summary_label'](config)})",
            timings,
            "norms_compare",
        ):
            passed = compare_norms(test_root, control_tarball, report_path, build_dir)
        bit_compare_status = "PASS" if passed else "FAIL"
        bit_compare_skip_reason = None

    total = time.time() - script_start

    if test_phase_failed:
        final_status, exit_code = "FAIL", 1
    elif control_status == "failed":
        final_status, exit_code = "INCONCLUSIVE", 2
    elif bit_compare_status == "PASS":
        final_status, exit_code = "PASS", 0
    else:
        final_status, exit_code = "FAIL", 1

    summary_lines = ci_lib.build_ci_summary(
        control_branch=config["control_branch"],
        test_branch=ci_lib.resolve_test_branch_label(config, __file__),
        control_status=control_status,
        control_build_status=control_stage_statuses["build"],
        control_test_status=control_stage_statuses["test"],
        test_build_status=test_stage_statuses["build"],
        test_test_status=test_stage_statuses["test"],
        control_tarball=control_tarball,
        bit_compare_status=bit_compare_status,
        bit_compare_skip_reason=bit_compare_skip_reason,
        final_status=final_status,
        report_path=report_path,
        timings=timings,
        total=total,
        timing_keys=("control-branch", "test-branch", "norms_compare"),
    )

    summary_lines = [
        summary_lines[0],
        summary_lines[1],
        summary_lines[2],
        f"  execution profile     : {profile['summary_label'](config)}",
        f"  CI banner             : {profile['banner']}",
        *summary_lines[3:],
    ]

    for line in summary_lines:
        logger.info(line)

    try:
        with open(report_path, "a", encoding="utf-8") as f:
            f.write("\n")
            for line in summary_lines:
                f.write(line + "\n")
    except OSError as e:
        logger.warning(f"Could not append CI summary to {report_path}: {e}")

    sys.exit(exit_code)