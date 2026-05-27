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
"""ECMWF-HPC CI entrypoint for OpenIFS."""

from host_ci_lib import run_profile


def main():
    run_profile("ecmwf_hpc")


if __name__ == "__main__":
    main()